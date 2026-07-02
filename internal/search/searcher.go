package search

import (
	"sort"
	"sync"
	"sync/atomic"

	"fable/internal/chess"
	"fable/internal/eval"
)

// Info is one "info" line worth of search progress for the UCI layer.
type Info struct {
	Depth     int
	SelDepth  int
	MultiPV   int // 1-based
	ScoreCP   int
	ScoreMate int // mate in N moves (signed); valid when IsMate
	IsMate    bool
	Bound     uint8 // BoundExact, or Lower/Upper for aspiration re-reports
	Nodes     int64
	NPS       int64
	TimeMs    int64
	Hashfull  int
	PV        []chess.Move
}

// Result is the final outcome of a search.
type Result struct {
	BestMove   chess.Move
	PonderMove chess.Move
	ScoreCP    int
	ScoreMate  int
	IsMate     bool
	Depth      int
	Nodes      int64
}

// Searcher owns the transposition table and worker threads. One search runs
// at a time; Stop and PonderHit may be called concurrently from another
// goroutine.
type Searcher struct {
	tt         *TT
	net        *eval.Network
	threadsN   int
	multiPV    int
	overheadMs int64

	stop      atomic.Bool
	pondering atomic.Bool
	tm        *timeManager
	lim       Limits

	threads []*thread
	onInfo  func(Info)
}

// NewSearcher creates a searcher with the given hash size (MiB) and thread
// count.
func NewSearcher(hashMB, threads, multiPV int) *Searcher {
	s := &Searcher{
		tt:         NewTT(hashMB),
		threadsN:   max(1, threads),
		multiPV:    max(1, multiPV),
		overheadMs: 30,
	}
	s.ensureThreads()
	return s
}

func (s *Searcher) ensureThreads() {
	for len(s.threads) < s.threadsN {
		s.threads = append(s.threads, newThread(len(s.threads), s))
	}
	if len(s.threads) > s.threadsN {
		s.threads = s.threads[:s.threadsN]
	}
}

func (s *Searcher) SetInfoHandler(f func(Info)) { s.onInfo = f }
func (s *Searcher) SetNet(n *eval.Network)      { s.net = n }
func (s *Searcher) SetHash(mb int)              { s.tt.Resize(mb) }
func (s *Searcher) SetMultiPV(n int)            { s.multiPV = max(1, n) }
func (s *Searcher) SetMoveOverhead(ms int)      { s.overheadMs = int64(ms) }
func (s *Searcher) Hashfull() int               { return s.tt.Hashfull() }

func (s *Searcher) SetThreads(n int) {
	s.threadsN = max(1, n)
	s.threads = nil
	s.ensureThreads()
}

// NewGame clears all state carried between searches.
func (s *Searcher) NewGame() {
	s.tt.Clear()
	for _, t := range s.threads {
		t.hist.clear()
	}
}

// Stop aborts the current search as soon as possible.
func (s *Searcher) Stop() {
	s.pondering.Store(false)
	s.stop.Store(true)
}

// PonderHit switches a ponder search to a normal timed search; the clock
// starts running now.
func (s *Searcher) PonderHit() {
	if s.tm != nil {
		s.tm.restart()
	}
	s.pondering.Store(false)
}

func (s *Searcher) totalNodes() int64 {
	var n int64
	for _, t := range s.threads {
		n += t.nodes.Load()
	}
	return n
}

// Search runs a blocking search on pos with the given limits and returns the
// best move. pos is not modified (each thread works on a clone).
func (s *Searcher) Search(pos *chess.Position, lim Limits) Result {
	s.lim = lim
	s.stop.Store(false)
	s.pondering.Store(lim.Ponder)
	s.tm = newTimeManager(lim, pos.Side(), s.overheadMs)
	s.tt.NewSearch()
	s.ensureThreads()

	pos.RootHist = pos.HistLen()

	// Build the root move list (optionally filtered by searchmoves).
	var ml chess.MoveList
	pos.GenLegal(&ml, false)
	rms := make([]rootMove, 0, ml.N)
	for i := 0; i < ml.N; i++ {
		m := ml.Moves[i]
		if len(lim.SearchMoves) > 0 {
			found := false
			for _, sm := range lim.SearchMoves {
				if sm == m {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		rms = append(rms, rootMove{move: m, score: -Inf, prevScore: -Inf})
	}
	if len(rms) == 0 && ml.N > 0 {
		// searchmoves filtered out everything; fall back to all legal moves.
		for i := 0; i < ml.N; i++ {
			rms = append(rms, rootMove{move: ml.Moves[i], score: -Inf, prevScore: -Inf})
		}
	}
	if len(rms) == 0 {
		return Result{BestMove: chess.NullMove} // mate or stalemate at root
	}

	// Initial ordering: captures/promotions by value, so that even an
	// instantly-stopped search returns a sane move.
	sort.SliceStable(rms, func(i, j int) bool {
		return rootOrderKey(pos, rms[i].move) > rootOrderKey(pos, rms[j].move)
	})

	// Prepare threads.
	for _, t := range s.threads {
		t.pos = pos.Clone()
		t.net = s.net
		t.rootMoves = make([]rootMove, len(rms))
		copy(t.rootMoves, rms)
		for i := range t.rootMoves {
			t.rootMoves[i].pv = nil
		}
		t.nodes.Store(0)
		t.seldepth = 0
		t.completedDepth = 0
		t.bestMoveChanges = 0
		t.checkCnt = 0
		for i := range t.ss {
			t.ss[i] = stackEntry{contIdx: -1}
		}
		if s.net != nil {
			s.net.Refresh(t.pos, &t.accs[0])
		}
	}

	var wg sync.WaitGroup
	for _, t := range s.threads[1:] {
		wg.Add(1)
		go func(t *thread) {
			defer wg.Done()
			t.iterate()
		}(t)
	}
	s.threads[0].iterate()
	s.stop.Store(true)
	wg.Wait()

	mt := s.threads[0]
	best := mt.rootMoves[0]

	res := Result{
		BestMove: best.move,
		Depth:    mt.completedDepth,
		Nodes:    s.totalNodes(),
	}
	score := best.score
	if score == -Inf {
		score = best.prevScore
	}
	res.ScoreCP, res.ScoreMate, res.IsMate = scoreForUCI(score)

	// Ponder move: second PV move, or a TT suggestion, always validated
	// against the legal moves of the position after the best move.
	res.PonderMove = s.ponderMoveFor(mt.pos, best)
	return res
}

func rootOrderKey(pos *chess.Position, m chess.Move) int {
	k := 0
	if pos.IsCapture(m) {
		victim := chess.Pawn
		if !m.IsEnPass() {
			victim = pos.PieceOn(m.To()).Type()
		}
		k = 1000 + chess.SEEValues[victim]
	}
	if m.IsPromo() {
		k += chess.SEEValues[m.PromoType()]
	}
	return k
}

func (s *Searcher) ponderMoveFor(pos *chess.Position, best rootMove) chess.Move {
	if best.move == chess.NullMove {
		return chess.NullMove
	}
	var cand chess.Move
	if len(best.pv) >= 2 {
		cand = best.pv[1]
	}
	pos.Make(best.move)
	defer pos.Unmake(best.move)
	if cand == chess.NullMove {
		if tte, ok := s.tt.Probe(pos.Key()); ok {
			cand = tte.Move
		}
	}
	if cand == chess.NullMove {
		return chess.NullMove
	}
	var ml chess.MoveList
	pos.GenLegal(&ml, false)
	if ml.Contains(cand) {
		return cand
	}
	return chess.NullMove
}

func scoreForUCI(score int) (cp, mate int, isMate bool) {
	if score >= MateBound {
		return 0, (Mate - score + 1) / 2, true
	}
	if score <= -MateBound {
		return 0, -((Mate + score + 1) / 2), true
	}
	return score, 0, false
}

// iterate runs iterative deepening with aspiration windows and MultiPV.
func (t *thread) iterate() {
	s := t.s
	maxDepth := MaxPly - 8
	if s.lim.Depth > 0 && s.lim.Depth < maxDepth {
		maxDepth = s.lim.Depth
	}

	for depth := 1; depth <= maxDepth; depth++ {
		t.rootDepth = depth
		for i := range t.rootMoves {
			t.rootMoves[i].prevScore = t.rootMoves[i].score
		}
		nPV := min(s.multiPV, len(t.rootMoves))

		for pvIdx := 0; pvIdx < nPV; pvIdx++ {
			alpha, beta := -Inf, Inf
			delta := 0
			prev := t.rootMoves[pvIdx].prevScore
			if depth >= 5 && prev > -Inf && abs(prev) < MateBound {
				delta = 18
				alpha = max(prev-delta, -Inf)
				beta = min(prev+delta, Inf)
			}

			for {
				score := t.rootSearch(depth, alpha, beta, pvIdx)
				sortRootMoves(t.rootMoves, pvIdx)
				if s.stop.Load() {
					break
				}
				if score <= alpha {
					beta = (alpha + beta) / 2
					alpha = max(score-delta, -Inf)
					if t.id == 0 && s.tm.elapsed().Milliseconds() > 3000 {
						s.reportPV(t, depth, pvIdx, BoundUpper)
					}
				} else if score >= beta {
					beta = min(score+delta, Inf)
					if t.id == 0 && s.tm.elapsed().Milliseconds() > 3000 {
						s.reportPV(t, depth, pvIdx, BoundLower)
					}
				} else {
					break
				}
				delta += delta/2 + 5
				if abs(score) >= MateBound {
					alpha, beta = -Inf, Inf
				}
			}
			if s.stop.Load() {
				break
			}
			if t.id == 0 {
				s.reportPV(t, depth, pvIdx, BoundExact)
			}
		}
		if s.stop.Load() {
			break
		}
		t.completedDepth = depth

		if t.id == 0 {
			bestScore := t.rootMoves[0].score
			if s.lim.Mate > 0 && bestScore >= Mate-2*s.lim.Mate {
				s.stop.Store(true)
				break
			}
			if !s.pondering.Load() {
				if s.lim.Nodes > 0 && s.totalNodes() >= s.lim.Nodes {
					s.stop.Store(true)
					break
				}
				stability := 1.0 + min(t.bestMoveChanges, 2.0)*0.6
				if s.tm.softExpired(stability) {
					s.stop.Store(true)
					break
				}
			}
			t.bestMoveChanges *= 0.6
		}
	}
}

// sortRootMoves stable-sorts rootMoves[from:] by (score, prevScore) desc.
func sortRootMoves(rms []rootMove, from int) {
	seg := rms[from:]
	sort.SliceStable(seg, func(i, j int) bool {
		if seg[i].score != seg[j].score {
			return seg[i].score > seg[j].score
		}
		return seg[i].prevScore > seg[j].prevScore
	})
}

// rootSearch searches root moves [pvIdx:] within (alpha, beta).
func (t *thread) rootSearch(depth, alpha, beta, pvIdx int) int {
	pos := t.pos
	if !pos.InCheck() {
		t.ss[0].staticEval = t.evaluate(0)
	} else {
		t.ss[0].staticEval = NoEval
	}
	t.ss[0].excluded = chess.NullMove

	bestScore := -Inf
	moveCount := 0

	for i := pvIdx; i < len(t.rootMoves); i++ {
		rm := &t.rootMoves[i]
		m := rm.move
		moveCount++
		isQuiet := pos.IsQuiet(m)

		info := eval.MoveInfoFor(pos, m)
		pos.Make(m)
		if t.net != nil {
			t.net.Update(&t.accs[0], &t.accs[1], info)
		}
		t.ss[0].currentMove = m
		t.ss[0].contIdx = pieceTo(info.Piece, m.To())

		newDepth := depth - 1
		var score int
		if moveCount == 1 {
			score = -t.search(newDepth, -beta, -alpha, 1, false)
		} else {
			r := 0
			if depth >= 3 && moveCount > 3 && isQuiet {
				r = int(lmrTable[min(depth, 63)][min(moveCount, 63)])
				if maxR := newDepth - 1; r > maxR {
					r = maxR
				}
				if r < 0 {
					r = 0
				}
			}
			score = -t.search(newDepth-r, -alpha-1, -alpha, 1, true)
			if score > alpha && r > 0 {
				score = -t.search(newDepth, -alpha-1, -alpha, 1, true)
			}
			if score > alpha {
				score = -t.search(newDepth, -beta, -alpha, 1, false)
			}
		}
		pos.Unmake(m)

		if t.s.stop.Load() {
			return 0
		}

		if moveCount == 1 || score > alpha {
			rm.score = score
			rm.pv = append(rm.pv[:0], m)
			rm.pv = append(rm.pv, t.ss[1].pv[:t.ss[1].pvLen]...)
			if pvIdx == 0 && moveCount > 1 {
				t.bestMoveChanges++
			}
		} else {
			rm.score = -Inf
		}

		if score > bestScore {
			bestScore = score
		}
		if score > alpha {
			alpha = score
			if alpha >= beta {
				break
			}
		}
	}
	return bestScore
}

func (s *Searcher) reportPV(t *thread, depth, pvIdx int, bound uint8) {
	if s.onInfo == nil {
		return
	}
	rm := &t.rootMoves[pvIdx]
	score := rm.score
	if score == -Inf {
		score = rm.prevScore
	}
	if score == -Inf {
		return
	}
	elapsed := s.tm.elapsed().Milliseconds()
	nodes := s.totalNodes()
	nps := int64(0)
	if elapsed > 0 {
		nps = nodes * 1000 / elapsed
	}
	info := Info{
		Depth:    depth,
		SelDepth: t.seldepth,
		MultiPV:  pvIdx + 1,
		Bound:    bound,
		Nodes:    nodes,
		NPS:      nps,
		TimeMs:   elapsed,
		Hashfull: s.tt.Hashfull(),
		PV:       rm.pv,
	}
	info.ScoreCP, info.ScoreMate, info.IsMate = scoreForUCI(score)
	s.onInfo(info)
}
