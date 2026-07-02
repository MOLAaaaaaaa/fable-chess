package eval

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"fable/internal/chess"
)

func randomNet(t *testing.T, hidden int, rng *rand.Rand) *Network {
	t.Helper()
	n, err := NewNetwork(hidden)
	if err != nil {
		t.Fatal(err)
	}
	for i := range n.FTWeights {
		n.FTWeights[i] = int16(rng.Intn(1011) - 505)
	}
	for i := range n.FTBias {
		n.FTBias[i] = int16(rng.Intn(1011) - 505)
	}
	for i := range n.OutWeights {
		n.OutWeights[i] = int16(rng.Intn(255) - 127)
	}
	n.OutBias = int32(rng.Intn(2000001) - 1000000)
	return n
}

// TestSIMDEquivalence verifies the dispatch kernels (AVX2 when available)
// against the pure Go reference on random data.
func TestSIMDEquivalence(t *testing.T) {
	t.Logf("HasAVX2 = %v", HasAVX2)
	rng := rand.New(rand.NewSource(42))
	for _, n := range []int{32, 64, 128, 256, 512} {
		mk := func() []int16 {
			v := make([]int16, n)
			for i := range v {
				v[i] = int16(rng.Intn(65536) - 32768)
			}
			return v
		}
		for trial := 0; trial < 50; trial++ {
			src, a1, a2, s1, s2 := mk(), mk(), mk(), mk(), mk()

			want := make([]int16, n)
			got := make([]int16, n)

			copy(want, src)
			copy(got, src)
			addGeneric(want, a1)
			addVec(got, a1)
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("add mismatch n=%d i=%d", n, i)
				}
			}

			addSubGeneric(want, src, a1, s1)
			addSubVec(got, src, a1, s1)
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("addSub mismatch n=%d i=%d", n, i)
				}
			}

			addSubSubGeneric(want, src, a1, s1, s2)
			addSubSubVec(got, src, a1, s1, s2)
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("addSubSub mismatch n=%d i=%d", n, i)
				}
			}

			addAddSubSubGeneric(want, src, a1, a2, s1, s2)
			addAddSubSubVec(got, src, a1, a2, s1, s2)
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("addAddSubSub mismatch n=%d i=%d", n, i)
				}
			}

			// Output layer with in-range weights.
			us, them := mk(), mk()
			w := make([]int16, 2*n)
			for i := range w {
				w[i] = int16(rng.Intn(255) - 127)
			}
			if g, wnt := outputVec(us, them, w), outputGeneric(us, them, w); g != wnt {
				t.Fatalf("output mismatch n=%d: got %d want %d", n, g, wnt)
			}
		}
	}
}

// TestIncrementalUpdate plays random legal games and verifies that the
// incrementally updated accumulator always matches a from-scratch refresh,
// covering quiets, captures, promotions, en passant and castling.
func TestIncrementalUpdate(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	net := randomNet(t, 64, rng)

	for game := 0; game < 30; game++ {
		pos := chess.MustPosition(chess.StartFEN)
		var accs [256]Accumulator
		net.Refresh(pos, &accs[0])
		var ml chess.MoveList
		for ply := 0; ply < 120; ply++ {
			pos.GenLegal(&ml, false)
			if ml.N == 0 {
				break
			}
			m := ml.Moves[rng.Intn(ml.N)]
			info := MoveInfoFor(pos, m)
			pos.Make(m)
			net.Update(&accs[ply], &accs[ply+1], info)

			var fresh Accumulator
			net.Refresh(pos, &fresh)
			h := net.Hidden
			for c := 0; c < 2; c++ {
				for i := 0; i < h; i++ {
					if fresh.vals[c][i] != accs[ply+1].vals[c][i] {
						t.Fatalf("game %d ply %d move %s: accumulator mismatch persp %d idx %d after %s",
							game, ply, m, c, i, pos.FEN())
					}
				}
			}
		}
	}
}

func TestSaveLoadRoundtrip(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	net := randomNet(t, 128, rng)
	path := filepath.Join(t.TempDir(), "test.nnue")
	if err := net.Save(path); err != nil {
		t.Fatal(err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Hidden != net.Hidden || got.OutBias != net.OutBias {
		t.Fatal("header mismatch")
	}
	for i := range net.FTWeights {
		if net.FTWeights[i] != got.FTWeights[i] {
			t.Fatal("ft weights mismatch")
		}
	}
	for i := range net.OutWeights {
		if net.OutWeights[i] != got.OutWeights[i] {
			t.Fatal("out weights mismatch")
		}
	}

	// Corrupt one byte: load must fail the checksum.
	data, _ := os.ReadFile(path)
	data[100] ^= 0xFF
	bad := filepath.Join(t.TempDir(), "bad.nnue")
	if err := os.WriteFile(bad, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(bad); err == nil {
		t.Fatal("corrupt file should fail to load")
	}
}

func TestEvalSymmetry(t *testing.T) {
	// A color-mirrored position must evaluate to the same score for the
	// mirrored side to move (the perspective architecture guarantees this).
	rng := rand.New(rand.NewSource(3))
	net := randomNet(t, 64, rng)

	fens := []struct{ a, b string }{
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"},
		{"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
			"rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3"},
	}
	for _, f := range fens {
		pa := chess.MustPosition(f.a)
		pb := chess.MustPosition(f.b)
		var aa, ab Accumulator
		net.Refresh(pa, &aa)
		net.Refresh(pb, &ab)
		if va, vb := net.Evaluate(pa, &aa), net.Evaluate(pb, &ab); va != vb {
			t.Fatalf("mirror asymmetry: %d vs %d for %s | %s", va, vb, f.a, f.b)
		}
	}
}
