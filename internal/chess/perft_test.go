package chess

import "testing"

// Standard perft positions from the Chess Programming Wiki. Together they
// exercise every tricky rule: castling legality, en passant pins and
// discovered checks, promotions (incl. underpromotion captures), double
// checks and pinned-piece movement.
var perftCases = []struct {
	fen    string
	counts []uint64 // counts[i] = perft(i+1)
}{
	{StartFEN,
		[]uint64{20, 400, 8902, 197281, 4865609, 119060324}},
	{"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete
		[]uint64{48, 2039, 97862, 4085603, 193690690}},
	{"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		[]uint64{14, 191, 2812, 43238, 674624, 11030083}},
	{"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
		[]uint64{6, 264, 9467, 422333, 15833292}},
	{"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
		[]uint64{44, 1486, 62379, 2103487, 89941194}},
	{"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
		[]uint64{46, 2079, 89890, 3894594, 164075551}},
}

func TestPerft(t *testing.T) {
	for _, tc := range perftCases {
		p := MustPosition(tc.fen)
		maxD := len(tc.counts)
		if testing.Short() {
			if maxD > 4 {
				maxD = 4
			}
		} else if maxD > 5 {
			maxD = 5
		}
		for d := 1; d <= maxD; d++ {
			if got := Perft(p, d); got != tc.counts[d-1] {
				div, _ := Divide(p, d)
				t.Fatalf("perft(%d) of %q = %d, want %d\ndivide:\n%s",
					d, tc.fen, got, tc.counts[d-1], div)
			}
		}
		// The position must be unchanged after perft (make/unmake symmetry).
		if p.FEN() != MustPosition(tc.fen).FEN() {
			t.Fatalf("position corrupted after perft: %q -> %q", tc.fen, p.FEN())
		}
	}
}

// TestMakeUnmakeConsistency walks a deep random-ish tree verifying that the
// incrementally maintained zobrist key, checkers bitboard and piece
// bitboards always match a from-scratch recomputation.
func TestMakeUnmakeConsistency(t *testing.T) {
	for _, tc := range perftCases {
		p := MustPosition(tc.fen)
		var walk func(depth int)
		var ml MoveList
		walk = func(depth int) {
			if p.key != p.computeKey() {
				t.Fatalf("incremental key mismatch at %s", p.FEN())
			}
			if p.checkers != p.computeCheckers() {
				t.Fatalf("checkers mismatch at %s", p.FEN())
			}
			var occ Bitboard
			for pt := Pawn; pt <= King; pt++ {
				occ |= p.byType[pt]
			}
			if occ != p.Occupied() || p.byColor[White]&p.byColor[Black] != 0 {
				t.Fatalf("bitboard inconsistency at %s", p.FEN())
			}
			for s := Square(0); s < 64; s++ {
				pc := p.board[s]
				if pc == NoPiece {
					if occ.IsSet(s) {
						t.Fatalf("mailbox/bitboard mismatch at %s sq %s", p.FEN(), s)
					}
					continue
				}
				if !p.PiecesOf(pc.Color(), pc.Type()).IsSet(s) {
					t.Fatalf("mailbox/bitboard mismatch at %s sq %s", p.FEN(), s)
				}
			}
			if depth == 0 {
				return
			}
			p.GenLegal(&ml, false)
			moves := make([]Move, ml.N)
			copy(moves, ml.Slice())
			for _, m := range moves {
				fenBefore := p.FEN()
				keyBefore := p.key
				p.Make(m)
				walk(depth - 1)
				p.Unmake(m)
				if p.FEN() != fenBefore || p.key != keyBefore {
					t.Fatalf("unmake mismatch after %s at %s: got %s", m, fenBefore, p.FEN())
				}
			}
		}
		walk(3)
	}
}

func TestEnPassantNormalization(t *testing.T) {
	// After 1.e4, black cannot capture en passant, so the ep square must be
	// normalized away and the key must equal the same position reached with
	// no ep square at all.
	p := MustPosition(StartFEN)
	m := p.ParseUCIMove("e2e4")
	if m == NullMove {
		t.Fatal("e2e4 should be legal")
	}
	p.Make(m)
	if p.EPSquare() != NoSquare {
		t.Fatalf("ep square should be normalized away, got %s", p.EPSquare())
	}
	q := MustPosition("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
	if p.Key() != q.Key() {
		t.Fatal("normalized ep position should hash identically")
	}

	// 1.e4 c5? no... construct real ep: 1.e4 a6 2.e5 d5 -> exd6 possible.
	p2 := MustPosition(StartFEN)
	for _, mv := range []string{"e2e4", "a7a6", "e4e5", "d7d5"} {
		m := p2.ParseUCIMove(mv)
		if m == NullMove {
			t.Fatalf("%s should be legal", mv)
		}
		p2.Make(m)
	}
	if p2.EPSquare() == NoSquare {
		t.Fatal("ep square should be set after d7d5 with white pawn on e5")
	}
	if p2.ParseUCIMove("e5d6") == NullMove {
		t.Fatal("en passant e5d6 should be legal")
	}
}

func TestRepetitionDetection(t *testing.T) {
	p := MustPosition(StartFEN)
	moves := []string{"g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"}
	for i, mv := range moves {
		m := p.ParseUCIMove(mv)
		if m == NullMove {
			t.Fatalf("move %s illegal", mv)
		}
		p.Make(m)
		isRep := p.IsRepetition()
		// After the 8th ply the start position occurs for the 3rd time.
		if i == len(moves)-1 && !isRep {
			t.Fatal("threefold repetition not detected")
		}
		if i < 3 && isRep {
			t.Fatalf("false repetition at ply %d", i+1)
		}
	}
}

func TestSEE(t *testing.T) {
	cases := []struct {
		fen  string
		mv   string
		thr  int
		want bool
	}{
		// Simple winning capture: pawn takes undefended pawn.
		{"k7/8/8/3p4/4P3/8/8/K7 w - - 0 1", "e4d5", 0, true},
		{"k7/8/8/3p4/4P3/8/8/K7 w - - 0 1", "e4d5", 100, true},
		{"k7/8/8/3p4/4P3/8/8/K7 w - - 0 1", "e4d5", 101, false},
		// Defended pawn taken by queen: loses queen for pawn.
		{"k7/8/4p3/3p4/8/8/3Q4/K7 w - - 0 1", "d2d5", 0, false},
		// Defended pawn taken by pawn: even trade.
		{"k7/8/4p3/3p4/4P3/8/8/K7 w - - 0 1", "e4d5", 0, true},
		// Rook takes pawn defended by rook, attacker x-ray backed by rook.
		{"k3r3/8/8/4p3/8/8/4R3/K3R3 w - - 0 1", "e2e5", 0, true},
		// Same but without the backup rook: loses the exchange.
		{"k3r3/8/8/4p3/8/8/4R3/K7 w - - 0 1", "e2e5", 0, false},
	}
	for _, tc := range cases {
		p := MustPosition(tc.fen)
		m := p.ParseUCIMove(tc.mv)
		if m == NullMove {
			t.Fatalf("%s illegal in %s", tc.mv, tc.fen)
		}
		if got := p.SEEGE(m, tc.thr); got != tc.want {
			t.Errorf("SEEGE(%s, %d) in %q = %v, want %v", tc.mv, tc.thr, tc.fen, got, tc.want)
		}
	}
}

func TestMagicsAgainstSlowGen(t *testing.T) {
	// Verify magic lookups equal the slow ray-walking reference on random
	// occupancies for every square.
	rng := &splitmix64{x: 12345}
	for s := Square(0); s < 64; s++ {
		for i := 0; i < 200; i++ {
			occ := Bitboard(rng.next() & rng.next())
			if RookAttacks(s, occ) != slidingAttackSlow(rookDeltas, s, occ) {
				t.Fatalf("rook attacks wrong at %s", s)
			}
			if BishopAttacks(s, occ) != slidingAttackSlow(bishopDeltas, s, occ) {
				t.Fatalf("bishop attacks wrong at %s", s)
			}
		}
	}
}

func BenchmarkPerftStart(b *testing.B) {
	p := MustPosition(StartFEN)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Perft(p, 4)
	}
}
