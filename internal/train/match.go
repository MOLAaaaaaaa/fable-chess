package train

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"fable/internal/chess"
	"fable/internal/eval"
	"fable/internal/search"
)

// MatchConfig controls an A-vs-B strength test.
type MatchConfig struct {
	Games   int // rounded up to pairs; each opening is played with both colors
	Nodes   int64
	Threads int
	NetA    string // "" or "none" = material bootstrap
	NetB    string
	Seed    uint64

	RandomPlies int
	MaxPly      int
	AdjWinCp    int
	AdjWinPlies int
	HashMB      int
}

func DefaultMatchConfig() MatchConfig {
	return MatchConfig{
		Games:       200,
		Nodes:       8000,
		Threads:     1,
		RandomPlies: 8,
		MaxPly:      300,
		AdjWinCp:    1200,
		AdjWinPlies: 4,
		HashMB:      16,
	}
}

func loadNetArg(path string) (*eval.Network, string, error) {
	if path == "" || path == "none" {
		return nil, "material", nil
	}
	n, err := eval.Load(path)
	if err != nil {
		return nil, "", err
	}
	return n, fmt.Sprintf("%s(h%d)", path, n.Hidden), nil
}

// RunMatch plays paired games (colors swapped per opening) and reports the
// score of A vs B with an Elo estimate.
func RunMatch(cfg MatchConfig) error {
	netA, nameA, err := loadNetArg(cfg.NetA)
	if err != nil {
		return err
	}
	netB, nameB, err := loadNetArg(cfg.NetB)
	if err != nil {
		return err
	}
	pairs := (cfg.Games + 1) / 2
	fmt.Printf("match: %s vs %s, %d games (%d paired openings), %d nodes/move\n",
		nameA, nameB, pairs*2, pairs, cfg.Nodes)

	var winA, winB, draws atomic.Int64
	var next atomic.Int64
	var wg sync.WaitGroup
	start := time.Now()

	worker := func() {
		defer wg.Done()
		sA := search.NewSearcher(cfg.HashMB, 1, 1)
		sA.SetNet(netA)
		sB := search.NewSearcher(cfg.HashMB, 1, 1)
		sB.SetNet(netB)
		for {
			p := next.Add(1) - 1
			if p >= int64(pairs) {
				return
			}
			opening := buildOpening(cfg, uint64(p))
			for swap := 0; swap < 2; swap++ {
				white, black := sA, sB
				aIsWhite := true
				if swap == 1 {
					white, black = sB, sA
					aIsWhite = false
				}
				res := playMatchGame(white, black, opening, cfg)
				switch {
				case res == 1:
					draws.Add(1)
				case (res == 2) == aIsWhite:
					winA.Add(1)
				default:
					winB.Add(1)
				}
			}
			done := (p + 1) * 2
			if done%50 == 0 {
				a, b, d := winA.Load(), winB.Load(), draws.Load()
				fmt.Printf("  %d games: A +%d =%d -%d (%.1f%%)\n",
					a+b+d, a, d, b, 100*score(a, d, b))
			}
		}
	}

	threads := max(1, cfg.Threads)
	for t := 0; t < threads; t++ {
		wg.Add(1)
		go worker()
	}
	wg.Wait()

	a, b, d := winA.Load(), winB.Load(), draws.Load()
	n := a + b + d
	p := score(a, d, b)
	elo, margin := eloEstimate(a, d, b)
	fmt.Printf("result: A +%d =%d -%d of %d  score %.1f%%  elo %+.0f +/- %.0f  (%.1fs)\n",
		a, d, b, n, 100*p, elo, margin, time.Since(start).Seconds())
	return nil
}

func score(w, d, l int64) float64 {
	n := w + d + l
	if n == 0 {
		return 0.5
	}
	return (float64(w) + 0.5*float64(d)) / float64(n)
}

func eloEstimate(w, d, l int64) (elo, margin float64) {
	n := float64(w + d + l)
	if n == 0 {
		return 0, 0
	}
	p := score(w, d, l)
	p = math.Min(math.Max(p, 1e-6), 1-1e-6)
	elo = -400 * math.Log10(1/p-1)
	// Wilson-ish margin from the game outcome variance.
	mean := p
	varSum := float64(w)*(1-mean)*(1-mean) + float64(d)*(0.5-mean)*(0.5-mean) + float64(l)*mean*mean
	sd := math.Sqrt(varSum/n) / math.Sqrt(n)
	lo := math.Min(math.Max(p-1.96*sd, 1e-6), 1-1e-6)
	hi := math.Min(math.Max(p+1.96*sd, 1e-6), 1-1e-6)
	margin = (-400*math.Log10(1/hi-1) + 400*math.Log10(1/lo-1)) / 2
	return elo, margin
}

func buildOpening(cfg MatchConfig, pairIdx uint64) []chess.Move {
	rng := &prng{x: cfg.Seed<<32 ^ (pairIdx+1)*0x9E3779B9}
	var ml chess.MoveList
	for {
		pos := chess.MustPosition(chess.StartFEN)
		moves := make([]chess.Move, 0, cfg.RandomPlies)
		ok := true
		for i := 0; i < cfg.RandomPlies; i++ {
			pos.GenLegal(&ml, false)
			if ml.N == 0 {
				ok = false
				break
			}
			m := ml.Moves[rng.intn(ml.N)]
			pos.Make(m)
			moves = append(moves, m)
		}
		if ok {
			return moves
		}
	}
}

// playMatchGame returns the white-POV result 0/1/2.
func playMatchGame(white, black *search.Searcher, opening []chess.Move, cfg MatchConfig) uint8 {
	pos := chess.MustPosition(chess.StartFEN)
	for _, m := range opening {
		pos.Make(m)
	}
	white.NewGame()
	black.NewGame()

	keyCount := map[uint64]int{pos.Key(): 1}
	winStreakW, winStreakB := 0, 0
	ply := len(opening)
	var ml chess.MoveList

	for {
		pos.GenLegal(&ml, false)
		if ml.N == 0 {
			if pos.InCheck() {
				if pos.Side() == chess.White {
					return 0
				}
				return 2
			}
			return 1
		}
		if pos.InsufficientMaterial() || pos.Rule50() >= 100 ||
			keyCount[pos.Key()] >= 3 || ply >= cfg.MaxPly {
			return 1
		}

		s := white
		if pos.Side() == chess.Black {
			s = black
		}
		res := s.Search(pos, search.Limits{Nodes: cfg.Nodes})
		if res.BestMove == chess.NullMove {
			return 1
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
		if winStreakW >= cfg.AdjWinPlies {
			return 2
		}
		if winStreakB >= cfg.AdjWinPlies {
			return 0
		}

		pos.Make(res.BestMove)
		keyCount[pos.Key()]++
		ply++
	}
}
