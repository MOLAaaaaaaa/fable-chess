package train

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"

	"fable/internal/chess"
	"fable/internal/eval"
)

// TrainConfig controls NNUE training.
type TrainConfig struct {
	Data        []string // glob patterns of record files
	Out         string
	Hidden      int
	Epochs      int
	BatchSize   int
	LR          float64
	Lambda      float64 // target = lambda*sigmoid(score/K) + (1-lambda)*result
	WeightDecay float64
	ValFrac     float64
	Patience    int // early-stopping patience in epochs (0 = off)
	Seed        int64
	Threads     int
	Dedup       bool
	Mirror      bool // horizontal-mirror data augmentation
}

func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		Hidden:      128,
		Epochs:      40,
		BatchSize:   8192,
		LR:          0.001,
		Lambda:      0.8,
		WeightDecay: 1e-5,
		ValFrac:     0.1,
		Patience:    6,
		Seed:        1,
		Threads:     runtime.NumCPU(),
		Dedup:       true,
		Mirror:      true,
	}
}

// dataset holds all samples in flat arrays for cache-friendly access.
type dataset struct {
	n         int
	featStart []int32  // n+1 offsets into wfeat/bfeat
	wfeat     []uint16 // white-perspective features
	bfeat     []uint16 // black-perspective features
	stm       []uint8
	scoreStm  []int16 // score from stm perspective
	resultStm []uint8 // 0 loss, 1 draw, 2 win, stm perspective
}

// mirrorFeature reflects a feature index horizontally (file a <-> h).
// Chess rules are left-right symmetric (castling is not encoded in the
// features), so mirrored samples are exact rule-symmetry augmentations that
// double the effective dataset — important with only a few thousand games.
// Note the vertical color-swap symmetry needs no augmentation: the
// dual-perspective architecture encodes it exactly.
func mirrorFeature(f uint16) uint16 { return f ^ 7 }

// buildDataset converts records to flat training arrays. With mirror set,
// every position is also added horizontally mirrored.
func buildDataset(recs []Record, mirror bool) *dataset {
	nSamples := len(recs)
	if mirror {
		nSamples *= 2
	}
	d := &dataset{
		n:         nSamples,
		featStart: make([]int32, 1, nSamples+1),
		stm:       make([]uint8, 0, nSamples),
		scoreStm:  make([]int16, 0, nSamples),
		resultStm: make([]uint8, 0, nSamples),
	}
	for i := range recs {
		r := &recs[i]
		start := len(d.wfeat)
		d.wfeat, d.bfeat = r.Features(d.wfeat, d.bfeat)
		d.featStart = append(d.featStart, int32(len(d.wfeat)))
		score, result := int(r.Score), r.Result
		if r.Stm == 1 { // black to move: flip to stm POV
			score = -score
			result = 2 - result
		}
		d.stm = append(d.stm, r.Stm)
		d.scoreStm = append(d.scoreStm, int16(score))
		d.resultStm = append(d.resultStm, result)

		if mirror {
			for _, f := range d.wfeat[start:len(d.wfeat):len(d.wfeat)] {
				d.wfeat = append(d.wfeat, mirrorFeature(f))
			}
			for _, f := range d.bfeat[start:len(d.bfeat):len(d.bfeat)] {
				d.bfeat = append(d.bfeat, mirrorFeature(f))
			}
			d.featStart = append(d.featStart, int32(len(d.wfeat)))
			d.stm = append(d.stm, r.Stm)
			d.scoreStm = append(d.scoreStm, int16(score))
			d.resultStm = append(d.resultStm, result)
		}
	}
	d.n = len(d.stm)
	return d
}

// floatNet is the float32 training model matching the quantized architecture.
type floatNet struct {
	hidden int
	ftW    []float32 // [768][hidden]
	ftB    []float32
	outW   []float32 // [2*hidden]
	outB   float32
}

func newFloatNet(hidden int, rng *rand.Rand) *floatNet {
	n := &floatNet{
		hidden: hidden,
		ftW:    make([]float32, eval.NumFeatures*hidden),
		ftB:    make([]float32, hidden),
		outW:   make([]float32, 2*hidden),
	}
	ftScale := float32(math.Sqrt(1.0 / 32.0)) // ~sqrt(1/avg active features)
	for i := range n.ftW {
		n.ftW[i] = (rng.Float32()*2 - 1) * ftScale * 0.5
	}
	outScale := float32(math.Sqrt(6.0 / float64(2*hidden)))
	for i := range n.outW {
		n.outW[i] = (rng.Float32()*2 - 1) * outScale
	}
	return n
}

func (n *floatNet) clone() *floatNet {
	c := &floatNet{hidden: n.hidden, outB: n.outB}
	c.ftW = append([]float32(nil), n.ftW...)
	c.ftB = append([]float32(nil), n.ftB...)
	c.outW = append([]float32(nil), n.outW...)
	return c
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

// forward computes the sigmoid-space prediction for sample i, optionally
// filling the gradient scratch buffers for the backward pass.
type scratch struct {
	accW, accB []float32
}

func (n *floatNet) forward(d *dataset, i int, sc *scratch) (pred float64) {
	h := n.hidden
	copy(sc.accW, n.ftB)
	copy(sc.accB, n.ftB)
	s, e := d.featStart[i], d.featStart[i+1]
	for _, f := range d.wfeat[s:e] {
		row := n.ftW[int(f)*h : int(f)*h+h]
		acc := sc.accW
		for j := range row {
			acc[j] += row[j]
		}
	}
	for _, f := range d.bfeat[s:e] {
		row := n.ftW[int(f)*h : int(f)*h+h]
		acc := sc.accB
		for j := range row {
			acc[j] += row[j]
		}
	}
	us, them := sc.accW, sc.accB
	if d.stm[i] == 1 {
		us, them = sc.accB, sc.accW
	}
	out := float64(n.outB)
	for j := 0; j < h; j++ {
		t := clamp01(us[j])
		out += float64(t * t * n.outW[j])
	}
	for j := 0; j < h; j++ {
		t := clamp01(them[j])
		out += float64(t * t * n.outW[h+j])
	}
	return sigmoid(out)
}

func clamp01(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// grads accumulates dense gradients.
type grads struct {
	ftW  []float32
	ftB  []float32
	outW []float32
	outB float32
}

func newGrads(hidden int) *grads {
	return &grads{
		ftW:  make([]float32, eval.NumFeatures*hidden),
		ftB:  make([]float32, hidden),
		outW: make([]float32, 2*hidden),
	}
}

func (g *grads) zero() {
	clearF32(g.ftW)
	clearF32(g.ftB)
	clearF32(g.outW)
	g.outB = 0
}

func clearF32(s []float32) {
	for i := range s {
		s[i] = 0
	}
}

// target returns the blended training target in sigmoid space.
func target(d *dataset, i int, lambda float64) float64 {
	ev := sigmoid(float64(d.scoreStm[i]) / float64(eval.Scale))
	res := float64(d.resultStm[i]) / 2.0
	return lambda*ev + (1-lambda)*res
}

// backward accumulates the gradient of (pred-target)^2 for sample i.
func (n *floatNet) backward(d *dataset, i int, lambda float64, sc *scratch, g *grads) float64 {
	h := n.hidden
	pred := n.forward(d, i, sc)
	tgt := target(d, i, lambda)
	diff := pred - tgt
	loss := diff * diff
	dOut := float32(2 * diff * pred * (1 - pred))

	us, them := sc.accW, sc.accB
	usIsWhite := d.stm[i] == 0
	if !usIsWhite {
		us, them = sc.accB, sc.accW
	}

	// Output layer gradients + accumulator deltas (reuse acc buffers).
	for j := 0; j < h; j++ {
		x := us[j]
		t := clamp01(x)
		g.outW[j] += dOut * t * t
		dAcc := float32(0)
		if x > 0 && x < 1 {
			dAcc = dOut * n.outW[j] * 2 * t
		}
		us[j] = dAcc
	}
	for j := 0; j < h; j++ {
		x := them[j]
		t := clamp01(x)
		g.outW[h+j] += dOut * t * t
		dAcc := float32(0)
		if x > 0 && x < 1 {
			dAcc = dOut * n.outW[h+j] * 2 * t
		}
		them[j] = dAcc
	}
	g.outB += dOut

	// us/them now hold dAccW/dAccB in perspective order; map back.
	dAccW, dAccB := us, them
	if !usIsWhite {
		dAccW, dAccB = them, us
	}

	s, e := d.featStart[i], d.featStart[i+1]
	for _, f := range d.wfeat[s:e] {
		row := g.ftW[int(f)*h : int(f)*h+h]
		for j := 0; j < h; j++ {
			row[j] += dAccW[j]
		}
	}
	for _, f := range d.bfeat[s:e] {
		row := g.ftW[int(f)*h : int(f)*h+h]
		for j := 0; j < h; j++ {
			row[j] += dAccB[j]
		}
	}
	for j := 0; j < h; j++ {
		g.ftB[j] += dAccW[j] + dAccB[j]
	}
	return loss
}

// adam holds optimizer state.
type adam struct {
	m, v  []float32 // one flat vector over all params
	t     int
	beta1 float64
	beta2 float64
	eps   float64
}

func newAdam(n int) *adam {
	return &adam{
		m: make([]float32, n), v: make([]float32, n),
		beta1: 0.9, beta2: 0.999, eps: 1e-8,
	}
}

// Train runs the full training pipeline and writes the quantized network.
func Train(cfg TrainConfig) error {
	if !eval.ValidHidden(cfg.Hidden) {
		return fmt.Errorf("train: invalid hidden size %d (multiple of 32, 32..%d)", cfg.Hidden, eval.MaxHidden)
	}
	recs, err := LoadRecords(cfg.Data)
	if err != nil {
		return err
	}
	fmt.Printf("train: loaded %d positions\n", len(recs))

	if cfg.Dedup {
		seen := make(map[uint64]struct{}, len(recs))
		out := recs[:0]
		for i := range recs {
			h := recs[i].Hash()
			if _, ok := seen[h]; ok {
				continue
			}
			seen[h] = struct{}{}
			out = append(out, recs[i])
		}
		fmt.Printf("train: %d after dedup (%.1f%% dropped)\n",
			len(out), 100*float64(len(recs)-len(out))/float64(max(1, len(recs))))
		recs = out
	}
	if len(recs) < 1000 {
		return fmt.Errorf("train: only %d positions — need more data", len(recs))
	}

	rng := rand.New(rand.NewSource(cfg.Seed))

	// Split train/validation BY GAME, not by position: positions from one
	// game are highly correlated, and a positionwise split lets the model
	// memorize game trajectories while still showing a flattering val loss.
	// With only a few thousand games this leakage is the main overfit trap.
	games := make(map[uint32][]int) // (fileIdx<<16 | gameID) -> record indices
	for i := range recs {
		key := uint32(recs[i].GameID) | uint32(recs[i].FileIdx)<<16
		games[key] = append(games[key], i)
	}
	gameKeys := make([]uint32, 0, len(games))
	for k := range games {
		gameKeys = append(gameKeys, k)
	}
	sort.Slice(gameKeys, func(i, j int) bool { return gameKeys[i] < gameKeys[j] })
	rng.Shuffle(len(gameKeys), func(i, j int) { gameKeys[i], gameKeys[j] = gameKeys[j], gameKeys[i] })

	wantVal := int(float64(len(recs)) * cfg.ValFrac)
	if wantVal < 100 {
		wantVal = min(100, len(recs)/10)
	}
	var valRecs, trainRecs []Record
	valGames := 0
	for _, k := range gameKeys {
		if len(valRecs) < wantVal {
			for _, i := range games[k] {
				valRecs = append(valRecs, recs[i])
			}
			valGames++
		} else {
			for _, i := range games[k] {
				trainRecs = append(trainRecs, recs[i])
			}
		}
	}
	rng.Shuffle(len(trainRecs), func(i, j int) { trainRecs[i], trainRecs[j] = trainRecs[j], trainRecs[i] })
	// Validation stays unmirrored so its loss remains comparable across runs.
	valSet := buildDataset(valRecs, false)
	trainSet := buildDataset(trainRecs, cfg.Mirror)
	fmt.Printf("train: game-wise split: %d/%d games -> validation\n", valGames, len(gameKeys))

	// With small datasets a large batch size starves Adam of update steps;
	// keep at least ~16 steps per epoch.
	batchSize := cfg.BatchSize
	if maxB := max(128, trainSet.n/16); batchSize > maxB {
		batchSize = maxB
		fmt.Printf("train: batch size reduced to %d for %d samples\n", batchSize, trainSet.n)
	}
	fmt.Printf("train: %d train / %d val, hidden=%d lambda=%.2f lr=%g batch=%d\n",
		trainSet.n, valSet.n, cfg.Hidden, cfg.Lambda, cfg.LR, batchSize)

	net := newFloatNet(cfg.Hidden, rng)
	nParams := len(net.ftW) + len(net.ftB) + len(net.outW) + 1
	opt := newAdam(nParams)

	threads := max(1, cfg.Threads)
	workerGrads := make([]*grads, threads)
	workerScratch := make([]*scratch, threads)
	for i := range workerGrads {
		workerGrads[i] = newGrads(cfg.Hidden)
		workerScratch[i] = &scratch{
			accW: make([]float32, cfg.Hidden),
			accB: make([]float32, cfg.Hidden),
		}
	}

	indices := make([]int, trainSet.n)
	for i := range indices {
		indices[i] = i
	}

	best := net.clone()
	bestVal := math.Inf(1)
	badEpochs := 0
	start := time.Now()

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		lr := cfg.LR
		if epoch > cfg.Epochs*6/10 {
			lr *= 0.1
		}
		if epoch > cfg.Epochs*17/20 {
			lr *= 0.1
		}

		rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		var trainLoss float64
		nBatches := 0
		for off := 0; off < len(indices); off += batchSize {
			endIdx := min(off+batchSize, len(indices))
			batch := indices[off:endIdx]
			loss := runBatch(net, trainSet, batch, cfg.Lambda, workerGrads, workerScratch)
			applyAdam(net, opt, workerGrads[0], lr, cfg.WeightDecay, len(batch))
			trainLoss += loss
			nBatches++
		}
		trainLoss /= float64(max(1, nBatches))

		valLoss := evalLoss(net, valSet, cfg.Lambda, workerScratch)
		marker := ""
		if valLoss < bestVal {
			bestVal = valLoss
			best = net.clone()
			badEpochs = 0
			marker = " *"
		} else {
			badEpochs++
		}
		fmt.Printf("epoch %3d/%d  lr %-8.2g train %.6f  val %.6f%s\n",
			epoch, cfg.Epochs, lr, trainLoss, valLoss, marker)
		if cfg.Patience > 0 && badEpochs >= cfg.Patience {
			fmt.Printf("early stop: no val improvement for %d epochs\n", cfg.Patience)
			break
		}
	}
	fmt.Printf("training time: %.1fs, best val loss %.6f\n", time.Since(start).Seconds(), bestVal)

	qnet, err := quantize(best)
	if err != nil {
		return err
	}
	verifyQuantization(best, qnet, valSet, workerScratch[0])
	if err := qnet.Save(cfg.Out); err != nil {
		return err
	}
	fmt.Printf("saved %s (hidden %d)\n", cfg.Out, qnet.Hidden)
	return nil
}

// runBatch computes gradients for a batch in parallel into workerGrads[0].
func runBatch(net *floatNet, d *dataset, batch []int, lambda float64,
	workerGrads []*grads, workerScratch []*scratch) float64 {

	threads := len(workerGrads)
	losses := make([]float64, threads)
	var wg sync.WaitGroup
	chunk := (len(batch) + threads - 1) / threads
	for w := 0; w < threads; w++ {
		lo := w * chunk
		if lo >= len(batch) {
			workerGrads[w].zero()
			continue
		}
		hi := min(lo+chunk, len(batch))
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			g := workerGrads[w]
			g.zero()
			var sum float64
			for _, idx := range batch[lo:hi] {
				sum += net.backward(d, idx, lambda, workerScratch[w], g)
			}
			losses[w] = sum
		}(w, lo, hi)
	}
	wg.Wait()

	// Reduce into workerGrads[0].
	g0 := workerGrads[0]
	for w := 1; w < threads; w++ {
		gw := workerGrads[w]
		addF32(g0.ftW, gw.ftW)
		addF32(g0.ftB, gw.ftB)
		addF32(g0.outW, gw.outW)
		g0.outB += gw.outB
	}
	var total float64
	for _, l := range losses {
		total += l
	}
	return total / float64(len(batch))
}

func addF32(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// applyAdam performs one AdamW step over all parameters.
func applyAdam(net *floatNet, opt *adam, g *grads, lr, wd float64, batchN int) {
	opt.t++
	bc1 := 1 - math.Pow(opt.beta1, float64(opt.t))
	bc2 := 1 - math.Pow(opt.beta2, float64(opt.t))
	scale := 1.0 / float64(batchN)

	off := 0
	step := func(params []float32, grads []float32, clampLo, clampHi float32) {
		for i := range params {
			gi := float64(grads[i]) * scale
			m := float64(opt.m[off+i])*opt.beta1 + (1-opt.beta1)*gi
			v := float64(opt.v[off+i])*opt.beta2 + (1-opt.beta2)*gi*gi
			opt.m[off+i] = float32(m)
			opt.v[off+i] = float32(v)
			upd := lr * (m / bc1) / (math.Sqrt(v/bc2) + opt.eps)
			p := float64(params[i]) - upd - lr*wd*float64(params[i])
			pf := float32(p)
			if pf < clampLo {
				pf = clampLo
			} else if pf > clampHi {
				pf = clampHi
			}
			params[i] = pf
		}
		off += len(params)
	}
	// Clamps keep every weight representable after quantization.
	step(net.ftW, g.ftW, -1.98, 1.98)
	step(net.ftB, g.ftB, -1.98, 1.98)
	step(net.outW, g.outW, -1.98, 1.98)
	ob := []float32{net.outB}
	gb := []float32{g.outB}
	step(ob, gb, -10, 10)
	net.outB = ob[0]
}

func evalLoss(net *floatNet, d *dataset, lambda float64, scratch []*scratch) float64 {
	threads := len(scratch)
	losses := make([]float64, threads)
	var wg sync.WaitGroup
	chunk := (d.n + threads - 1) / threads
	for w := 0; w < threads; w++ {
		lo := w * chunk
		if lo >= d.n {
			continue
		}
		hi := min(lo+chunk, d.n)
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			var sum float64
			for i := lo; i < hi; i++ {
				diff := net.forward(d, i, scratch[w]) - target(d, i, lambda)
				sum += diff * diff
			}
			losses[w] = sum
		}(w, lo, hi)
	}
	wg.Wait()
	var total float64
	for _, l := range losses {
		total += l
	}
	return total / float64(max(1, d.n))
}

// quantize converts the float model to the engine's int16 format.
func quantize(net *floatNet) (*eval.Network, error) {
	q, err := eval.NewNetwork(net.hidden)
	if err != nil {
		return nil, err
	}
	roundI16 := func(x float64, lo, hi int) int16 {
		v := int(math.RoundToEven(x))
		if v < lo {
			v = lo
		} else if v > hi {
			v = hi
		}
		return int16(v)
	}
	for i, w := range net.ftW {
		q.FTWeights[i] = roundI16(float64(w)*eval.QA, -32000, 32000)
	}
	for i, b := range net.ftB {
		q.FTBias[i] = roundI16(float64(b)*eval.QA, -32000, 32000)
	}
	for i, w := range net.outW {
		q.OutWeights[i] = roundI16(float64(w)*eval.QB, -127, 127)
	}
	ob := math.RoundToEven(float64(net.outB) * eval.QA * eval.QA * eval.QB)
	q.OutBias = int32(ob)
	return q, nil
}

// verifyQuantization reports the cp error between the float model and the
// quantized network evaluated through the exact engine arithmetic.
func verifyQuantization(fn *floatNet, qn *eval.Network, d *dataset, sc *scratch) {
	n := min(d.n, 2000)
	var sumAbs float64
	maxAbs := 0.0
	for i := 0; i < n; i++ {
		pred := fn.forward(d, i, sc)
		// Convert sigmoid-space prediction back to cp.
		p := math.Min(math.Max(pred, 1e-9), 1-1e-9)
		floatCP := math.Log(p/(1-p)) * float64(eval.Scale)

		s, e := d.featStart[i], d.featStart[i+1]
		stm := chess.Color(d.stm[i])
		qcp := float64(qn.EvalFeatures(d.wfeat[s:e], d.bfeat[s:e], stm))

		diff := math.Abs(floatCP - qcp)
		sumAbs += diff
		if diff > maxAbs {
			maxAbs = diff
		}
	}
	fmt.Printf("quantization check (%d samples): mean |err| %.2f cp, max %.2f cp\n",
		n, sumAbs/float64(max(1, n)), maxAbs)
}
