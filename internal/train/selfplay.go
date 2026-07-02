package train

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"fable/internal/chess"
	"fable/internal/eval"
	"fable/internal/search"
)

// SelfplayConfig controls self-play data generation.
type SelfplayConfig struct {
	Games   int
	Nodes   int64  // soft node limit per move
	Threads int    // parallel games
	NetPath string // "" or "none" -> material bootstrap (generation 0)
	Out     string
	Append  bool
	Seed    uint64

	RandomPlies    int // random opening plies (alternates +1 by game parity)
	OpeningCutoff  int // reject openings with |eval| above this, cp
	AdjWinCp       int // adjudicate win when |score| stays above this
	AdjWinPlies    int
	AdjDrawCp      int
	AdjDrawPlies   int
	AdjDrawMinPly  int
	MaxPly         int
	MinSavePly     int
	SaveScoreLimit int // don't save positions with |score| above this
	HashMB         int
}

// DefaultSelfplayConfig returns sensible defaults for small runs.
func DefaultSelfplayConfig() SelfplayConfig {
	return SelfplayConfig{
		Games:          1000,
		Nodes:          8000,
		Threads:        1,
		RandomPlies:    8,
		OpeningCutoff:  250,
		AdjWinCp:       1200,
		AdjWinPlies:    4,
		AdjDrawCp:      8,
		AdjDrawPlies:   12,
		AdjDrawMinPly:  90,
		MaxPly:         300,
		MinSavePly:     14,
		SaveScoreLimit: 3000,
		HashMB:         16,
	}
}

type spStats struct {
	games     atomic.Int64
	positions atomic.Int64
	whiteWins atomic.Int64
	blackWins atomic.Int64
	draws     atomic.Int64
	plies     atomic.Int64
}

type prng struct{ x uint64 }

func (r *prng) next() uint64 {
	r.x += 0x9E3779B97F4A7C15
	z := r.x
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

func (r *prng) intn(n int) int { return int(r.next() % uint64(n)) }

// RunSelfplay plays cfg.Games self-play games and appends the filtered
// positions to cfg.Out.
func RunSelfplay(cfg SelfplayConfig) error {
	var net *eval.Network
	if cfg.NetPath != "" && cfg.NetPath != "none" {
		n, err := eval.Load(cfg.NetPath)
		if err != nil {
			return fmt.Errorf("selfplay: %w", err)
		}
		net = n
		fmt.Printf("selfplay: using net %s (hidden %d)\n", cfg.NetPath, n.Hidden)
	} else {
		fmt.Println("selfplay: generation 0, material bootstrap eval")
	}

	w, err := newRecordWriter(cfg.Out, cfg.Append)
	if err != nil {
		return err
	}

	stats := &spStats{}
	recCh := make(chan []Record, 64)
	writeErr := make(chan error, 1)
	go func() {
		for recs := range recCh {
			if err := w.writeAll(recs); err != nil {
				select {
				case writeErr <- err:
				default:
				}
				return
			}
		}
		writeErr <- w.close()
	}()

	var next atomic.Int64
	var wg sync.WaitGroup
	threads := max(1, cfg.Threads)
	start := time.Now()

	for t := 0; t < threads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := search.NewSearcher(cfg.HashMB, 1, 1)
			s.SetNet(net)
			for {
				g := next.Add(1) - 1
				if g >= int64(cfg.Games) {
					return
				}
				recs, result, plies := playGame(s, cfg, uint64(g))
				stats.games.Add(1)
				stats.positions.Add(int64(len(recs)))
				stats.plies.Add(int64(plies))
				switch result {
				case 2:
					stats.whiteWins.Add(1)
				case 0:
					stats.blackWins.Add(1)
				default:
					stats.draws.Add(1)
				}
				if len(recs) > 0 {
					recCh <- recs
				}
			}
		}()
	}

	progressDone := make(chan struct{})
	go func() {
		tick := time.NewTicker(5 * time.Second)
		defer tick.Stop()
		for {
			select {
			case <-progressDone:
				return
			case <-tick.C:
				g := stats.games.Load()
				p := stats.positions.Load()
				el := time.Since(start).Seconds()
				fmt.Printf("selfplay: %d/%d games, %d positions, +%d =%d -%d, %.1f games/s\n",
					g, cfg.Games, p, stats.whiteWins.Load(), stats.draws.Load(),
					stats.blackWins.Load(), float64(g)/el)
			}
		}
	}()

	wg.Wait()
	close(progressDone)
	close(recCh)
	if err := <-writeErr; err != nil {
		return err
	}

	g := stats.games.Load()
	avgPly := float64(0)
	if g > 0 {
		avgPly = float64(stats.plies.Load()) / float64(g)
	}
	fmt.Printf("selfplay done: %d games (%d W / %d D / %d L white-POV), %d positions, avg %.0f plies, %.1fs\n",
		g, stats.whiteWins.Load(), stats.draws.Load(), stats.blackWins.Load(),
		stats.positions.Load(), avgPly, time.Since(start).Seconds())
	fmt.Printf("output: %s\n", cfg.Out)
	_ = os.Stdout.Sync()
	return nil
}

// playGame plays one game. Returns saved records, the white-POV result
// (0/1/2) and the number of plies played.
func playGame(s *search.Searcher, cfg SelfplayConfig, gameIdx uint64) ([]Record, uint8, int) {
	rng := &prng{x: cfg.Seed<<32 ^ (gameIdx+1)*0x1000193}
	randomPlies := cfg.RandomPlies + int(gameIdx&1)

	var pos *chess.Position
	var ml chess.MoveList

	// Build a random balanced opening.
	for attempt := 0; ; attempt++ {
		pos = chess.MustPosition(chess.StartFEN)
		ok := true
		for i := 0; i < randomPlies; i++ {
			pos.GenLegal(&ml, false)
			if ml.N == 0 {
				ok = false
				break
			}
			pos.Make(ml.Moves[rng.intn(ml.N)])
		}
		if !ok {
			continue
		}
		if attempt >= 20 {
			break
		}
		s.NewGame()
		res := s.Search(pos, search.Limits{Nodes: cfg.Nodes})
		if res.BestMove == chess.NullMove || res.IsMate {
			continue
		}
		cp := res.ScoreCP
		if abs(cp) <= cfg.OpeningCutoff {
			break
		}
	}
	s.NewGame()

	records := make([]Record, 0, 128)
	keyCount := map[uint64]int{pos.Key(): 1}
	result := uint8(1)
	winStreakW, winStreakB, drawStreak := 0, 0, 0
	ply := randomPlies

	for {
		pos.GenLegal(&ml, false)
		if ml.N == 0 {
			if pos.InCheck() {
				if pos.Side() == chess.White {
					result = 0
				} else {
					result = 2
				}
			} else {
				result = 1 // stalemate
			}
			break
		}
		if pos.InsufficientMaterial() || pos.Rule50() >= 100 ||
			keyCount[pos.Key()] >= 3 || ply >= cfg.MaxPly {
			result = 1
			break
		}

		res := s.Search(pos, search.Limits{Nodes: cfg.Nodes})
		if res.BestMove == chess.NullMove {
			result = 1
			break
		}

		scoreWhite := res.ScoreCP
		if res.IsMate {
			scoreWhite = 32000
			if res.ScoreMate < 0 {
				scoreWhite = -32000
			}
		}
		if pos.Side() == chess.Black {
			scoreWhite = -scoreWhite
		}

		// Save filter: quiet, out of opening, not in check, sane score.
		if ply >= cfg.MinSavePly && !pos.InCheck() && !res.IsMate &&
			abs(scoreWhite) <= cfg.SaveScoreLimit &&
			pos.IsQuiet(res.BestMove) {
			records = append(records, RecordFromPosition(pos, scoreWhite, ply))
		}

		// Adjudication.
		switch {
		case scoreWhite >= cfg.AdjWinCp:
			winStreakW++
			winStreakB = 0
		case scoreWhite <= -cfg.AdjWinCp:
			winStreakB++
			winStreakW = 0
		default:
			winStreakW, winStreakB = 0, 0
		}
		if abs(scoreWhite) <= cfg.AdjDrawCp && ply >= cfg.AdjDrawMinPly {
			drawStreak++
		} else {
			drawStreak = 0
		}
		if winStreakW >= cfg.AdjWinPlies {
			result = 2
			break
		}
		if winStreakB >= cfg.AdjWinPlies {
			result = 0
			break
		}
		if drawStreak >= cfg.AdjDrawPlies {
			result = 1
			break
		}

		pos.Make(res.BestMove)
		keyCount[pos.Key()]++
		ply++
	}

	// GameID mixes the seed so that appended runs with different seeds don't
	// systematically collide. A residual collision merges two games into one
	// split group — harmless (conservative) for train/val separation.
	gid := uint16((&prng{x: cfg.Seed*0x9E3779B9 + gameIdx + 1}).next())
	for i := range records {
		records[i].Result = result
		records[i].GameID = gid
	}
	return records, result, ply
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
