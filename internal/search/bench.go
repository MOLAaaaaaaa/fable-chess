package search

import (
	"fmt"
	"io"
	"time"

	"fable/internal/chess"
)

// benchFENs is a small mixed suite (openings, middlegames, endgames,
// tactics) used by the "bench" command for speed and reproducibility checks.
var benchFENs = []string{
	chess.StartFEN,
	"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
	"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
	"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
	"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
	"r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 1 9",
	"8/8/1p1k4/p1p2p2/P1P2P2/1P1K4/8/8 w - - 0 1",
	"8/pp3k2/5np1/2p1p3/2P4P/1P2PP2/P5K1/8 b - - 0 1",
	"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
	"2rq1rk1/pb2bppp/1p2pn2/8/2pP4/2N1PN2/PPQ1BPPP/2R2RK1 w - - 0 12",
	"2r3k1/pp3ppp/8/8/8/1P6/P4PPP/3R2K1 w - - 0 1",
	"6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
}

// Bench searches every bench position to the given depth and reports total
// nodes and speed. Returns total nodes (stable for a fixed build/config).
func Bench(depth int, w io.Writer) int64 {
	if depth <= 0 {
		depth = 12
	}
	s := NewSearcher(16, 1, 1)
	var total int64
	start := time.Now()
	for i, fen := range benchFENs {
		pos, err := chess.NewPosition(fen)
		if err != nil {
			panic("bench: bad FEN " + fen + ": " + err.Error())
		}
		s.NewGame()
		res := s.Search(pos, Limits{Depth: depth})
		total += res.Nodes
		fmt.Fprintf(w, "position %2d/%d  bestmove %-6s nodes %10d\n",
			i+1, len(benchFENs), res.BestMove, res.Nodes)
	}
	elapsed := time.Since(start)
	nps := int64(0)
	if ms := elapsed.Milliseconds(); ms > 0 {
		nps = total * 1000 / ms
	}
	fmt.Fprintf(w, "\nTime  : %d ms\nNodes : %d\nNPS   : %d\n", elapsed.Milliseconds(), total, nps)
	fmt.Fprintf(w, "Nodes searched: %d\n", total)
	return total
}
