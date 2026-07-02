package search

import (
	"io"
	"testing"

	"fable/internal/chess"
)

func TestFindsMate(t *testing.T) {
	s := NewSearcher(16, 1, 1)
	// 1.Kb3 Kb1 2.Re1#
	pos := chess.MustPosition("8/8/8/8/8/2K5/4R3/k7 w - - 0 1")
	res := s.Search(pos, Limits{Depth: 12})
	if !res.IsMate || res.ScoreMate != 2 {
		t.Fatalf("want mate 2, got mate=%v score=%d cp=%d", res.IsMate, res.ScoreMate, res.ScoreCP)
	}
	if res.BestMove.String() != "c3b3" {
		t.Fatalf("want c3b3, got %s", res.BestMove)
	}
}

func TestMatedPosition(t *testing.T) {
	s := NewSearcher(16, 1, 1)
	// Black is checkmated (Ra8#, king h6 covers escape squares).
	pos := chess.MustPosition("R6k/8/7K/8/8/8/8/8 b - - 0 1")
	res := s.Search(pos, Limits{Depth: 5})
	if res.BestMove != chess.NullMove {
		t.Fatalf("mated position must yield null best move, got %s", res.BestMove)
	}
	// Stalemate likewise.
	pos = chess.MustPosition("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1")
	res = s.Search(pos, Limits{Depth: 5})
	if res.BestMove != chess.NullMove {
		t.Fatalf("stalemate must yield null best move, got %s", res.BestMove)
	}
}

// TestFiftyMovePreference: white is a queen down but any rook move reaches
// the 50-move draw. The search must score the position as a draw.
func TestFiftyMovePreference(t *testing.T) {
	s := NewSearcher(16, 1, 1)
	pos := chess.MustPosition("kq6/8/8/8/8/8/8/K6R w - - 99 50")
	res := s.Search(pos, Limits{Depth: 10})
	if res.IsMate || res.ScoreCP != 0 {
		t.Fatalf("want draw score 0, got mate=%v cp=%d (move %s)", res.IsMate, res.ScoreCP, res.BestMove)
	}
}

// TestRepetitionScore: down a rook, the side to move can force repetition
// via perpetual check; the search should find a ~0 score, not a big deficit.
func TestRepetitionAwareness(t *testing.T) {
	s := NewSearcher(16, 1, 1)
	// White queen checks forever between e6/g4 mirrors: classic perpetual.
	pos := chess.MustPosition("6k1/5p1p/6p1/8/8/6r1/5PPP/3Q2K1 w - - 0 1")
	res := s.Search(pos, Limits{Depth: 14})
	// White is down a rook; a perpetual (Qd5+/Qd8+...) holds a draw.
	if res.ScoreCP < -150 {
		t.Fatalf("white should hold near-draw via checks, got cp=%d best=%s", res.ScoreCP, res.BestMove)
	}
}

func TestSearchMultiThreadRace(t *testing.T) {
	s := NewSearcher(32, 4, 2)
	pos := chess.MustPosition("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
	res := s.Search(pos, Limits{Depth: 10})
	if res.BestMove == chess.NullMove {
		t.Fatal("no best move")
	}
	var ml chess.MoveList
	pos.GenLegal(&ml, false)
	if !ml.Contains(res.BestMove) {
		t.Fatalf("illegal best move %s", res.BestMove)
	}
}

func TestBenchSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	if n := Bench(6, io.Discard); n <= 0 {
		t.Fatal("bench produced no nodes")
	}
}
