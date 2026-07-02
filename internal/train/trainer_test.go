package train

import (
	"math"
	"math/rand"
	"testing"

	"fable/internal/chess"
	"fable/internal/eval"
)

// tiny synthetic dataset: random sparse positions with score = material.
func synthRecords(n int, rng *rand.Rand) []Record {
	recs := make([]Record, 0, n)
	values := [6]int{100, 300, 300, 500, 900, 0}
	for len(recs) < n {
		pos := chess.MustPosition(chess.StartFEN)
		var ml chess.MoveList
		plies := 4 + rng.Intn(60)
		ok := true
		for i := 0; i < plies; i++ {
			pos.GenLegal(&ml, false)
			if ml.N == 0 {
				ok = false
				break
			}
			pos.Make(ml.Moves[rng.Intn(ml.N)])
		}
		if !ok || pos.InCheck() {
			continue
		}
		mat := 0
		for pt := chess.Pawn; pt <= chess.Queen; pt++ {
			mat += values[pt] * (pos.PiecesOf(chess.White, pt).Count() -
				pos.PiecesOf(chess.Black, pt).Count())
		}
		r := RecordFromPosition(pos, mat, plies)
		switch {
		case mat > 150:
			r.Result = 2
		case mat < -150:
			r.Result = 0
		default:
			r.Result = 1
		}
		recs = append(recs, r)
	}
	return recs
}

// TestBackwardNumericalGradient verifies the analytic gradients against
// finite differences on a handful of parameters.
func TestBackwardNumericalGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	recs := synthRecords(4, rng)
	d := buildDataset(recs, false)
	const h = 32
	net := newFloatNet(h, rng)
	// Give biases some non-zero values so clamp boundaries are exercised.
	for i := range net.ftB {
		net.ftB[i] = (rng.Float32()*2 - 1) * 0.1
	}
	net.outB = 0.05

	sc := &scratch{accW: make([]float32, h), accB: make([]float32, h)}
	g := newGrads(h)
	lambda := 0.7
	for i := 0; i < d.n; i++ {
		net.backward(d, i, lambda, sc, g)
	}

	lossAll := func() float64 {
		var sum float64
		for i := 0; i < d.n; i++ {
			diff := net.forward(d, i, sc) - target(d, i, lambda)
			sum += diff * diff
		}
		return sum
	}

	checks := 0
	check := func(param *float32, analytic float32, name string) {
		old := *param
		const eps = 1e-3
		*param = old + eps
		lp := lossAll()
		*param = old - eps
		lm := lossAll()
		*param = old
		numeric := (lp - lm) / (2 * eps)
		if math.Abs(numeric) < 1e-7 && math.Abs(float64(analytic)) < 1e-7 {
			return
		}
		rel := math.Abs(numeric-float64(analytic)) /
			math.Max(1e-6, math.Max(math.Abs(numeric), math.Abs(float64(analytic))))
		if rel > 0.02 {
			t.Errorf("%s: numeric %.8f analytic %.8f (rel %.3f)", name, numeric, analytic, rel)
		}
		checks++
	}

	// Sample a spread of parameters, favoring ones with non-zero gradient.
	for k := 0; k < 4000 && checks < 60; k++ {
		i := rng.Intn(len(net.ftW))
		if g.ftW[i] != 0 {
			check(&net.ftW[i], g.ftW[i], "ftW")
		}
	}
	for i := 0; i < h; i++ {
		check(&net.ftB[i], g.ftB[i], "ftB")
	}
	for i := 0; i < 2*h; i += 3 {
		check(&net.outW[i], g.outW[i], "outW")
	}
	check(&net.outB, g.outB, "outB")
	if checks < 30 {
		t.Fatalf("too few effective gradient checks: %d", checks)
	}
}

// TestLearnsMaterial trains a tiny net on synthetic material-labeled data
// and verifies that it learns the signal.
func TestLearnsMaterial(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	recs := synthRecords(6000, rng)
	d := buildDataset(recs, false)

	net := newFloatNet(32, rng)
	opt := newAdam(len(net.ftW) + len(net.ftB) + len(net.outW) + 1)
	sc := &scratch{accW: make([]float32, 32), accB: make([]float32, 32)}
	g := newGrads(32)

	baseline := evalLoss(net, d, 0.9, []*scratch{sc})
	idx := make([]int, d.n)
	for i := range idx {
		idx[i] = i
	}
	for epoch := 0; epoch < 30; epoch++ {
		rng.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
		for off := 0; off < len(idx); off += 1024 {
			end := min(off+1024, len(idx))
			g.zero()
			for _, i := range idx[off:end] {
				net.backward(d, i, 0.9, sc, g)
			}
			applyAdam(net, opt, g, 0.002, 0, end-off)
		}
	}
	final := evalLoss(net, d, 0.9, []*scratch{sc})
	t.Logf("loss %.6f -> %.6f", baseline, final)
	if final > baseline*0.35 {
		t.Fatalf("net failed to learn material: loss %.6f -> %.6f", baseline, final)
	}

	// A queen-up position (well out of the training distribution) must still
	// evaluate clearly positive for the side to move.
	qn, err := quantize(net)
	if err != nil {
		t.Fatal(err)
	}
	pos := chess.MustPosition("k7/8/8/8/8/8/8/KQ6 w - - 0 1")
	var acc eval.Accumulator
	qn.Refresh(pos, &acc)
	if cp := qn.Evaluate(pos, &acc); cp < 100 {
		t.Fatalf("queen-up eval too low: %d cp", cp)
	}
}
