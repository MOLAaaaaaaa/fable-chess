// Package uci implements the Universal Chess Interface protocol:
// position/go/stop/ponderhit handling, options (Hash, Threads, MultiPV,
// Ponder, Move Overhead, EvalFile), MultiPV info output and pondering
// semantics (bestmove is withheld until stop/ponderhit).
package uci

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"fable/internal/chess"
	"fable/internal/eval"
	"fable/internal/search"
)

const (
	EngineName   = "Fable"
	EngineVer    = "1.0"
	EngineAuthor = "Claude (Anthropic)"

	defaultNetFile = "fable.nnue"
)

type searchCtl struct {
	release     chan struct{} // closed on stop or ponderhit
	releaseOnce sync.Once
	done        chan struct{} // closed after bestmove is printed
	waiting     atomic.Bool   // search may need release before bestmove
}

func (c *searchCtl) releaseNow() {
	c.releaseOnce.Do(func() { close(c.release) })
}

func (c *searchCtl) released() bool {
	select {
	case <-c.release:
		return true
	default:
		return false
	}
}

// Engine is one UCI session.
type Engine struct {
	in  *bufio.Scanner
	out io.Writer
	mu  sync.Mutex // serializes all output lines

	searcher *search.Searcher
	pos      *chess.Position
	net      *eval.Network
	netPath  string

	hash     int
	threads  int
	multiPV  int
	overhead int

	cur *searchCtl
}

func New(in io.Reader, out io.Writer) *Engine {
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<22)
	e := &Engine{
		in:       sc,
		out:      out,
		hash:     64,
		threads:  1,
		multiPV:  1,
		overhead: 30,
		netPath:  defaultNetFile,
	}
	e.searcher = search.NewSearcher(e.hash, e.threads, e.multiPV)
	e.searcher.SetInfoHandler(e.printInfo)
	e.pos = chess.MustPosition(chess.StartFEN)
	e.loadNet(false)
	return e
}

func (e *Engine) send(format string, args ...any) {
	e.mu.Lock()
	fmt.Fprintf(e.out, format+"\n", args...)
	if f, ok := e.out.(interface{ Flush() error }); ok {
		_ = f.Flush()
	}
	e.mu.Unlock()
}

// findNetFile resolves the EvalFile path: as given (absolute or relative to
// the working directory), then next to the executable.
func findNetFile(path string) string {
	if path == "" {
		return ""
	}
	if _, err := os.Stat(path); err == nil {
		return path
	}
	if exe, err := os.Executable(); err == nil {
		p := filepath.Join(filepath.Dir(exe), path)
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func (e *Engine) loadNet(verbose bool) {
	resolved := findNetFile(e.netPath)
	if resolved == "" {
		e.net = nil
		e.searcher.SetNet(nil)
		if verbose {
			e.send("info string NNUE file %q not found, using material evaluation", e.netPath)
		}
		return
	}
	net, err := eval.Load(resolved)
	if err != nil {
		e.net = nil
		e.searcher.SetNet(nil)
		e.send("info string NNUE load failed (%v), using material evaluation", err)
		return
	}
	e.net = net
	e.searcher.SetNet(net)
	simd := "generic"
	if eval.HasAVX2 {
		simd = "avx2"
	}
	e.send("info string NNUE loaded: %s (768x2->%dx2->1 SCReLU, %s)", resolved, net.Hidden, simd)
}

// Run processes UCI commands until quit/EOF.
func Run(in io.Reader, out io.Writer) {
	e := New(in, out)
	for e.in.Scan() {
		line := strings.TrimSpace(e.in.Text())
		if line == "" {
			continue
		}
		if !e.handle(line) {
			break
		}
	}
	e.stopAndWait()
}

func (e *Engine) handle(line string) bool {
	fields := strings.Fields(line)
	cmd := fields[0]
	args := fields[1:]

	switch cmd {
	case "uci":
		e.send("id name %s %s", EngineName, EngineVer)
		e.send("id author %s", EngineAuthor)
		e.send("option name Hash type spin default 64 min 1 max 65536")
		e.send("option name Threads type spin default 1 min 1 max 256")
		e.send("option name MultiPV type spin default 1 min 1 max 256")
		e.send("option name Ponder type check default false")
		e.send("option name Move Overhead type spin default 30 min 0 max 5000")
		e.send("option name EvalFile type string default %s", defaultNetFile)
		e.send("option name Clear Hash type button")
		e.send("uciok")
	case "isready":
		e.send("readyok")
	case "setoption":
		e.setOption(args)
	case "ucinewgame":
		e.stopAndWait()
		e.searcher.NewGame()
	case "position":
		e.stopAndWait()
		e.setPosition(args)
	case "go":
		e.stopAndWait()
		e.goCmd(args)
	case "stop":
		e.stopCmd()
	case "ponderhit":
		e.ponderhitCmd()
	case "quit":
		return false
	case "debug", "register":
		// accepted, no-op
	case "d":
		e.send("%s", e.pos.String())
	case "eval":
		var acc eval.Accumulator
		if e.net != nil {
			e.net.Refresh(e.pos, &acc)
		}
		e.send("info string static eval (stm) = %d cp", eval.Evaluate(e.pos, e.net, &acc))
	case "bench":
		e.stopAndWait()
		depth := 12
		if len(args) > 0 {
			if d, err := strconv.Atoi(args[0]); err == nil {
				depth = d
			}
		}
		e.mu.Lock()
		search.Bench(depth, e.out)
		e.mu.Unlock()
	case "perft":
		depth := 5
		if len(args) > 0 {
			if d, err := strconv.Atoi(args[0]); err == nil {
				depth = d
			}
		}
		div, total := chess.Divide(e.pos, depth)
		e.send("%stotal %d", div, total)
		_ = total
	default:
		// Unknown commands are ignored per the UCI specification.
	}
	return true
}

func (e *Engine) searching() bool {
	if e.cur == nil {
		return false
	}
	select {
	case <-e.cur.done:
		e.cur = nil
		return false
	default:
		return true
	}
}

// stopAndWait stops any running search and waits for its bestmove.
func (e *Engine) stopAndWait() {
	if !e.searching() {
		return
	}
	e.searcher.Stop()
	e.cur.releaseNow()
	<-e.cur.done
	e.cur = nil
}

func (e *Engine) stopCmd() {
	if !e.searching() {
		return
	}
	e.searcher.Stop()
	e.cur.releaseNow()
}

func (e *Engine) ponderhitCmd() {
	if !e.searching() {
		return
	}
	e.searcher.PonderHit()
	e.cur.releaseNow()
}

func (e *Engine) setOption(args []string) {
	// setoption name <name...> [value <value...>]
	name, value := "", ""
	mode := ""
	for _, tok := range args {
		switch strings.ToLower(tok) {
		case "name":
			mode = "name"
		case "value":
			mode = "value"
		default:
			if mode == "name" {
				if name != "" {
					name += " "
				}
				name += tok
			} else if mode == "value" {
				if value != "" {
					value += " "
				}
				value += tok
			}
		}
	}
	e.stopAndWait()

	atoi := func(def, lo, hi int) int {
		v, err := strconv.Atoi(value)
		if err != nil {
			return def
		}
		if v < lo {
			v = lo
		}
		if v > hi {
			v = hi
		}
		return v
	}

	switch strings.ToLower(name) {
	case "hash":
		e.hash = atoi(e.hash, 1, 65536)
		e.searcher.SetHash(e.hash)
	case "threads":
		e.threads = atoi(e.threads, 1, 256)
		e.searcher.SetThreads(e.threads)
	case "multipv":
		e.multiPV = atoi(e.multiPV, 1, 256)
		e.searcher.SetMultiPV(e.multiPV)
	case "move overhead":
		e.overhead = atoi(e.overhead, 0, 5000)
		e.searcher.SetMoveOverhead(e.overhead)
	case "evalfile":
		e.netPath = value
		if e.netPath == "" {
			e.netPath = defaultNetFile
		}
		e.loadNet(true)
	case "clear hash":
		e.searcher.NewGame()
	case "ponder":
		// Pondering is driven by the GUI via "go ponder"; nothing to set.
	}
}

func (e *Engine) setPosition(args []string) {
	i := 0
	var pos *chess.Position
	var err error
	switch {
	case i < len(args) && args[i] == "startpos":
		pos = chess.MustPosition(chess.StartFEN)
		i++
	case i < len(args) && args[i] == "fen":
		i++
		start := i
		for i < len(args) && args[i] != "moves" {
			i++
		}
		pos, err = chess.NewPosition(strings.Join(args[start:i], " "))
		if err != nil {
			e.send("info string invalid fen: %v", err)
			return
		}
	default:
		e.send("info string invalid position command")
		return
	}
	if i < len(args) && args[i] == "moves" {
		i++
		for ; i < len(args); i++ {
			m := pos.ParseUCIMove(args[i])
			if m == chess.NullMove {
				e.send("info string illegal move in position command: %s", args[i])
				break
			}
			pos.Make(m)
		}
	}
	e.pos = pos
}

func (e *Engine) goCmd(args []string) {
	var lim search.Limits
	geti64 := func(i int) int64 {
		if i < len(args) {
			if v, err := strconv.ParseInt(args[i], 10, 64); err == nil {
				return v
			}
		}
		return 0
	}
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "wtime":
			i++
			lim.WTime = geti64(i)
		case "btime":
			i++
			lim.BTime = geti64(i)
		case "winc":
			i++
			lim.WInc = geti64(i)
		case "binc":
			i++
			lim.BInc = geti64(i)
		case "movestogo":
			i++
			lim.MovesToGo = int(geti64(i))
		case "depth":
			i++
			lim.Depth = int(geti64(i))
		case "nodes":
			i++
			lim.Nodes = geti64(i)
		case "mate":
			i++
			lim.Mate = int(geti64(i))
		case "movetime":
			i++
			lim.MoveTime = geti64(i)
		case "infinite":
			lim.Infinite = true
		case "ponder":
			lim.Ponder = true
		case "searchmoves":
			for i+1 < len(args) {
				m := e.pos.ParseUCIMove(args[i+1])
				if m == chess.NullMove {
					break
				}
				lim.SearchMoves = append(lim.SearchMoves, m)
				i++
			}
		}
	}

	ctl := &searchCtl{
		release: make(chan struct{}),
		done:    make(chan struct{}),
	}
	e.cur = ctl
	pos := e.pos
	holdBest := lim.Ponder || lim.Infinite

	go func() {
		defer close(ctl.done)
		res := e.searcher.Search(pos, lim)
		if holdBest && !ctl.released() {
			// UCI: in ponder/infinite mode bestmove must wait for
			// stop/ponderhit even if the search finished on its own.
			<-ctl.release
		}
		if res.BestMove == chess.NullMove {
			e.send("bestmove 0000")
			return
		}
		// Final legality guard: the move must be legal in the root position.
		var ml chess.MoveList
		pos.GenLegal(&ml, false)
		if !ml.Contains(res.BestMove) {
			if ml.N == 0 {
				e.send("bestmove 0000")
				return
			}
			res.BestMove = ml.Moves[0]
			res.PonderMove = chess.NullMove
		}
		if res.PonderMove != chess.NullMove {
			e.send("bestmove %s ponder %s", res.BestMove, res.PonderMove)
		} else {
			e.send("bestmove %s", res.BestMove)
		}
	}()
}

func (e *Engine) printInfo(info search.Info) {
	var sb strings.Builder
	fmt.Fprintf(&sb, "info depth %d seldepth %d multipv %d score ",
		info.Depth, info.SelDepth, info.MultiPV)
	if info.IsMate {
		fmt.Fprintf(&sb, "mate %d", info.ScoreMate)
	} else {
		fmt.Fprintf(&sb, "cp %d", info.ScoreCP)
	}
	switch info.Bound {
	case search.BoundLower:
		sb.WriteString(" lowerbound")
	case search.BoundUpper:
		sb.WriteString(" upperbound")
	}
	fmt.Fprintf(&sb, " nodes %d nps %d hashfull %d time %d pv",
		info.Nodes, info.NPS, info.Hashfull, info.TimeMs)
	for _, m := range info.PV {
		sb.WriteByte(' ')
		sb.WriteString(m.String())
	}
	e.send("%s", sb.String())
}
