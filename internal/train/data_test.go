package train

import (
	"math/rand"
	"path/filepath"
	"testing"

	"fable/internal/chess"
	"fable/internal/eval"
)

// TestRecordRoundtrip verifies Marshal/Unmarshal identity and that the
// feature extraction from a record matches a fresh position refresh through
// the quantized engine arithmetic.
func TestRecordRoundtrip(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	// Random-quantized net for cross-checking evals.
	net, err := eval.NewNetwork(32)
	if err != nil {
		t.Fatal(err)
	}
	for i := range net.FTWeights {
		net.FTWeights[i] = int16(rng.Intn(511) - 255)
	}
	for i := range net.FTBias {
		net.FTBias[i] = int16(rng.Intn(511) - 255)
	}
	for i := range net.OutWeights {
		net.OutWeights[i] = int16(rng.Intn(255) - 127)
	}

	var ml chess.MoveList
	for game := 0; game < 10; game++ {
		pos := chess.MustPosition(chess.StartFEN)
		for ply := 0; ply < 80; ply++ {
			pos.GenLegal(&ml, false)
			if ml.N == 0 {
				break
			}
			pos.Make(ml.Moves[rng.Intn(ml.N)])

			r := RecordFromPosition(pos, 123, ply)
			r.Result = 2
			r.GameID = uint16(game)

			// Serialization roundtrip.
			var buf [RecordSize]byte
			r.Marshal(buf[:])
			got := UnmarshalRecord(buf[:])
			got.FileIdx = r.FileIdx
			if got != r {
				t.Fatalf("roundtrip mismatch at game %d ply %d", game, ply)
			}

			// Feature-eval vs position-eval consistency.
			var wf, bf []uint16
			wf, bf = r.Features(wf, bf)
			cpA := net.EvalFeatures(wf, bf, chess.Color(r.Stm))
			var acc eval.Accumulator
			net.Refresh(pos, &acc)
			cpB := net.Evaluate(pos, &acc)
			if cpA != cpB {
				t.Fatalf("feature eval %d != refresh eval %d at %s", cpA, cpB, pos.FEN())
			}

			// FEN reconstruction must parse back to the same piece placement.
			p2, err := chess.NewPosition(r.FEN())
			if err != nil {
				t.Fatalf("record FEN invalid %q: %v", r.FEN(), err)
			}
			for s := chess.Square(0); s < 64; s++ {
				if p2.PieceOn(s) != pos.PieceOn(s) {
					t.Fatalf("FEN board mismatch at %s", s)
				}
			}
		}
	}
}

// TestWriterReaderFiles verifies the file writer/reader path incl. append.
func TestWriterReaderFiles(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "d.bin")
	rng := rand.New(rand.NewSource(4))
	recs := synthRecords(300, rng)
	for i := range recs {
		recs[i].GameID = uint16(i / 30)
	}

	w, err := newRecordWriter(path, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.writeAll(recs[:200]); err != nil {
		t.Fatal(err)
	}
	if err := w.close(); err != nil {
		t.Fatal(err)
	}
	w2, err := newRecordWriter(path, true) // append
	if err != nil {
		t.Fatal(err)
	}
	if err := w2.writeAll(recs[200:]); err != nil {
		t.Fatal(err)
	}
	if err := w2.close(); err != nil {
		t.Fatal(err)
	}

	got, err := LoadRecords([]string{path})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(recs) {
		t.Fatalf("got %d records, want %d", len(got), len(recs))
	}
	for i := range got {
		want := recs[i]
		want.FileIdx = 0
		if got[i] != want {
			t.Fatalf("record %d mismatch", i)
		}
	}
}
