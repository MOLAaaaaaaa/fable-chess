package search

import (
	"math"
	"sync/atomic"

	"fable/internal/chess"
	"fable/internal/eval"
)

type stackEntry struct {
	pv          [MaxPly + 1]chess.Move
	pvLen       int
	killers     [2]chess.Move
	staticEval  int
	currentMove chess.Move
	contIdx     int // piece*64+to of the move made at this ply; -1 for null/none
	excluded    chess.Move
}

type rootMove struct {
	move      chess.Move
	score     int
	prevScore int
	pv        []chess.Move
}

type thread struct {
	id  int
	s   *Searcher
	pos *chess.Position
	net *eval.Network

	accs []eval.Accumulator
	ss   []stackEntry
	hist *history

	nodes    atomic.Int64
	seldepth int

	rootDepth       int
	rootMoves       []rootMove
	completedDepth  int
	bestMoveChanges float64
	checkCnt        int
}

func newThread(id int, s *Searcher) *thread {
	return &thread{
		id:   id,
		s:    s,
		accs: make([]eval.Accumulator, MaxPly+8),
		ss:   make([]stackEntry, MaxPly+8),
		hist: &history{},
	}
}

var lmrTable [64][64]int8

func init() {
	for d := 1; d < 64; d++ {
		for m := 1; m < 64; m++ {
			lmrTable[d][m] = int8(0.8 + math.Log(float64(d))*math.Log(float64(m))/2.25)
		}
	}
}

func (t *thread) evaluate(ply int) int {
	return eval.Evaluate(t.pos, t.net, &t.accs[ply])
}

func (t *thread) updatePV(ply int, m chess.Move) {
	child := &t.ss[ply+1]
	e := &t.ss[ply]
	e.pv[0] = m
	copy(e.pv[1:1+child.pvLen], child.pv[:child.pvLen])
	e.pvLen = child.pvLen + 1
}

// periodicCheck enforces hard time and node limits (main thread only).
func (t *thread) periodicCheck() {
	t.checkCnt++
	mask := 511
	if t.s.lim.Nodes > 0 {
		mask = 127
	}
	if t.checkCnt&mask != 0 {
		return
	}
	s := t.s
	if s.lim.Nodes > 0 && s.totalNodes() >= s.lim.Nodes && !s.pondering.Load() {
		s.stop.Store(true)
	}
	if s.pondering.Load() {
		return
	}
	if t.completedDepth >= 1 && s.tm.hardExpired() {
		s.stop.Store(true)
	}
}

func rfpMargin(depth int, improving bool) int {
	m := 80 * depth
	if improving {
		m -= 55
	}
	return m
}

func lmpLimit(depth int, improving bool) int {
	n := 3 + depth*depth
	if !improving {
		n /= 2
	}
	return n
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

// search is the main PVS alpha-beta recursion. Called with ply >= 1.
func (t *thread) search(depth, alpha, beta, ply int, cutNode bool) int {
	t.ss[ply].pvLen = 0

	if depth <= 0 {
		return t.qsearch(alpha, beta, ply)
	}

	if t.s.stop.Load() {
		return 0
	}
	t.nodes.Add(1)
	if t.id == 0 {
		t.periodicCheck()
	}

	pos := t.pos
	inCheck := pos.InCheck()
	pvNode := beta-alpha > 1

	if ply > t.seldepth {
		t.seldepth = ply
	}
	if ply >= MaxPly-1 {
		if inCheck {
			return DrawScore
		}
		return t.evaluate(ply)
	}

	// Draw detection (before TT: the TT must never mask a rule draw).
	if pos.InsufficientMaterial() || pos.IsRepetition() {
		return DrawScore
	}
	if pos.Rule50() >= 100 {
		if !inCheck || pos.HasLegalMoves() {
			return DrawScore
		}
		// In check with no legal moves on the 50-move boundary: checkmate,
		// detected below by the empty move list.
	}

	// Mate distance pruning.
	if a := -Mate + ply; alpha < a {
		alpha = a
	}
	if b := Mate - ply - 1; beta > b {
		beta = b
	}
	if alpha >= beta {
		return alpha
	}

	excluded := t.ss[ply].excluded

	// Transposition table.
	var tte TTEntry
	ttHit := false
	if excluded == chess.NullMove {
		tte, ttHit = t.s.tt.Probe(pos.Key())
	}
	ttMove := chess.NullMove
	ttScore := 0
	if ttHit {
		ttMove = tte.Move
		ttScore = scoreFromTT(tte.Score, ply)
	}
	if !pvNode && ttHit && tte.Depth >= depth && pos.Rule50() < 90 {
		switch tte.Bound {
		case BoundExact:
			return ttScore
		case BoundLower:
			if ttScore >= beta {
				return ttScore
			}
		case BoundUpper:
			if ttScore <= alpha {
				return ttScore
			}
		}
	}

	// Static evaluation.
	rawEval := NoEval
	evalV := NoEval
	if !inCheck {
		if ttHit && tte.Eval != NoEval {
			rawEval = tte.Eval
		} else {
			rawEval = t.evaluate(ply)
		}
		evalV = rawEval
		// A TT score with a compatible bound is a better eval estimate.
		if ttHit {
			if tte.Bound == BoundExact ||
				(tte.Bound == BoundLower && ttScore > evalV) ||
				(tte.Bound == BoundUpper && ttScore < evalV) {
				evalV = ttScore
			}
		}
	}
	t.ss[ply].staticEval = rawEval

	improving := !inCheck && ply >= 2 &&
		t.ss[ply-2].staticEval != NoEval && rawEval > t.ss[ply-2].staticEval

	// Internal iterative reduction: no TT move at high depth.
	if depth >= 4 && ttMove == chess.NullMove && excluded == chess.NullMove {
		depth--
	}

	// Whole-node pruning (non-PV, not in check, not in a singular search).
	if !pvNode && !inCheck && excluded == chess.NullMove {
		// Reverse futility pruning.
		if depth <= 8 && evalV < MateBound && evalV-rfpMargin(depth, improving) >= beta {
			return evalV
		}
		// Null move pruning.
		if depth >= 3 && evalV >= beta &&
			t.ss[ply-1].currentMove != chess.NullMove &&
			pos.NonPawnMaterial(pos.Side()) {
			R := 3 + depth/3 + min((evalV-beta)/200, 3)
			pos.MakeNull()
			if t.net != nil {
				t.net.Copy(&t.accs[ply+1], &t.accs[ply])
			}
			t.ss[ply].currentMove = chess.NullMove
			t.ss[ply].contIdx = -1
			score := -t.search(depth-R, -beta, -beta+1, ply+1, !cutNode)
			pos.UnmakeNull()
			if t.s.stop.Load() {
				return 0
			}
			if score >= beta {
				if score >= MateBound {
					score = beta // unproven mates from null search are not trusted
				}
				return score
			}
		}
	}

	var mp movePicker
	mp.init(t, ttMove, ply, false)

	if mp.list.N == 0 {
		if excluded != chess.NullMove {
			return alpha
		}
		if inCheck {
			return -Mate + ply
		}
		return DrawScore // stalemate
	}

	// Clear the children's killers so they stay local to their subtree.
	t.ss[ply+1].killers[0] = chess.NullMove
	t.ss[ply+1].killers[1] = chess.NullMove

	bestScore := -Inf
	bestMove := chess.NullMove
	moveCount := 0
	skipQuiets := false
	var quietsTried [64]chess.Move
	nQuiets := 0
	var capsTried [32]chess.Move
	nCaps := 0

	for {
		m := mp.next(skipQuiets)
		if m == chess.NullMove {
			break
		}
		if m == excluded {
			continue
		}
		isQuiet := pos.IsQuiet(m)
		moveCount++
		histScore := 0
		if isQuiet {
			histScore = t.quietScore(m, ply) // must be read before Make
		}

		// Shallow-depth pruning, only after a real score is on the board.
		if !pvNode && bestScore > -MateBound {
			if isQuiet {
				if depth <= 8 && !inCheck && moveCount > lmpLimit(depth, improving) {
					skipQuiets = true
					continue
				}
				if depth <= 8 && !inCheck && evalV != NoEval &&
					evalV+100+120*depth <= alpha {
					skipQuiets = true
					continue
				}
				if depth <= 8 && !pos.SEEGE(m, -40*depth) {
					continue
				}
			} else if depth <= 8 && !pos.SEEGE(m, -100*depth) {
				continue
			}
		}

		// Singular extension: is the TT move much better than everything else?
		ext := 0
		if m == ttMove && excluded == chess.NullMove && depth >= 8 &&
			tte.Bound&BoundLower != 0 && abs(ttScore) < MateBound &&
			tte.Depth >= depth-3 && ply < 2*t.rootDepth {
			sBeta := ttScore - 2*depth
			t.ss[ply].excluded = m
			sScore := t.search((depth-1)/2, sBeta-1, sBeta, ply, cutNode)
			t.ss[ply].excluded = chess.NullMove
			t.ss[ply].pvLen = 0
			if sScore < sBeta {
				ext = 1
			} else if sBeta >= beta {
				return sBeta // multi-cut: even without the TT move we beat beta
			}
		}

		info := eval.MoveInfoFor(pos, m)
		pos.Make(m)
		if t.net != nil {
			t.net.Update(&t.accs[ply], &t.accs[ply+1], info)
		}
		t.ss[ply].currentMove = m
		t.ss[ply].contIdx = pieceTo(info.Piece, m.To())

		givesCheck := pos.InCheck()
		if ext == 0 && givesCheck && ply < 2*t.rootDepth {
			ext = 1
		}
		newDepth := depth - 1 + ext

		var score int
		if moveCount == 1 {
			score = -t.search(newDepth, -beta, -alpha, ply+1, false)
		} else {
			// Late move reductions.
			r := 0
			if depth >= 3 && moveCount > 2+btoi(pvNode) && isQuiet {
				r = int(lmrTable[min(depth, 63)][min(moveCount, 63)])
				if pvNode {
					r--
				}
				if !improving {
					r++
				}
				if cutNode {
					r++
				}
				if givesCheck {
					r--
				}
				// Reduce well-scoring quiets less, poor ones more.
				r -= max(-2, min(2, histScore/8192))
				if r < 0 {
					r = 0
				}
				if maxR := newDepth - 1; r > maxR {
					r = maxR
				}
			}
			score = -t.search(newDepth-r, -alpha-1, -alpha, ply+1, true)
			if score > alpha && r > 0 {
				score = -t.search(newDepth, -alpha-1, -alpha, ply+1, !cutNode)
			}
			if score > alpha && pvNode {
				score = -t.search(newDepth, -beta, -alpha, ply+1, false)
			}
		}
		pos.Unmake(m)

		if t.s.stop.Load() {
			return 0
		}

		if score > bestScore {
			bestScore = score
			if score > alpha {
				bestMove = m
				if pvNode {
					t.updatePV(ply, m)
				}
				if score >= beta {
					break
				}
				alpha = score
			}
		}

		if isQuiet {
			if nQuiets < len(quietsTried) {
				quietsTried[nQuiets] = m
				nQuiets++
			}
		} else if nCaps < len(capsTried) {
			capsTried[nCaps] = m
			nCaps++
		}
	}

	if bestScore == -Inf {
		// Only possible when the sole legal move was excluded (singular search).
		return alpha
	}

	// History updates on a beta cutoff.
	if bestMove != chess.NullMove && bestScore >= beta {
		bonus := histBonus(depth)
		if pos.IsQuiet(bestMove) {
			t.updateQuietHist(bestMove, bonus, ply)
			k := &t.ss[ply].killers
			if k[0] != bestMove {
				k[1] = k[0]
				k[0] = bestMove
			}
			if ply >= 1 {
				if idx := t.ss[ply-1].contIdx; idx >= 0 {
					t.hist.counter[idx] = bestMove
				}
			}
			for i := 0; i < nQuiets; i++ {
				if quietsTried[i] != bestMove {
					t.updateQuietHist(quietsTried[i], -bonus, ply)
				}
			}
		} else {
			t.updateCaptureHist(bestMove, bonus)
		}
		for i := 0; i < nCaps; i++ {
			if capsTried[i] != bestMove {
				t.updateCaptureHist(capsTried[i], -bonus)
			}
		}
	}

	if excluded == chess.NullMove {
		bound := BoundUpper
		if bestScore >= beta {
			bound = BoundLower
		} else if pvNode && bestMove != chess.NullMove {
			bound = BoundExact
		}
		t.s.tt.Store(pos.Key(), bestMove, scoreToTT(bestScore, ply), rawEval, depth, bound)
	}

	return bestScore
}

// qsearch resolves tactical sequences: captures and promotions only, or all
// evasions when in check.
func (t *thread) qsearch(alpha, beta, ply int) int {
	if t.s.stop.Load() {
		return 0
	}
	t.nodes.Add(1)
	if t.id == 0 {
		t.periodicCheck()
	}

	pos := t.pos
	inCheck := pos.InCheck()
	pvNode := beta-alpha > 1

	if ply > t.seldepth {
		t.seldepth = ply
	}

	if pos.InsufficientMaterial() || pos.IsRepetition() {
		return DrawScore
	}
	if pos.Rule50() >= 100 {
		if !inCheck || pos.HasLegalMoves() {
			return DrawScore
		}
	}
	if ply >= MaxPly-1 {
		if inCheck {
			return DrawScore
		}
		return t.evaluate(ply)
	}

	tte, ttHit := t.s.tt.Probe(pos.Key())
	ttMove := chess.NullMove
	ttScore := 0
	if ttHit {
		ttMove = tte.Move
		ttScore = scoreFromTT(tte.Score, ply)
	}
	if !pvNode && ttHit && pos.Rule50() < 90 {
		switch tte.Bound {
		case BoundExact:
			return ttScore
		case BoundLower:
			if ttScore >= beta {
				return ttScore
			}
		case BoundUpper:
			if ttScore <= alpha {
				return ttScore
			}
		}
	}

	bestScore := -Inf
	rawEval := NoEval
	if !inCheck {
		if ttHit && tte.Eval != NoEval {
			rawEval = tte.Eval
		} else {
			rawEval = t.evaluate(ply)
		}
		bestScore = rawEval
		if ttHit {
			if tte.Bound == BoundExact ||
				(tte.Bound == BoundLower && ttScore > bestScore) ||
				(tte.Bound == BoundUpper && ttScore < bestScore) {
				bestScore = ttScore
			}
		}
		if bestScore >= beta {
			return bestScore
		}
		if bestScore > alpha {
			alpha = bestScore
		}
	}

	var mp movePicker
	mp.init(t, ttMove, ply, true)

	if inCheck && mp.list.N == 0 {
		return -Mate + ply
	}

	bestMove := chess.NullMove
	for {
		m := mp.next(false)
		if m == chess.NullMove {
			break
		}
		if !inCheck {
			// Skip losing captures entirely.
			if !pos.SEEGE(m, 0) {
				continue
			}
			// Delta pruning: even winning the victim can't lift alpha.
			if rawEval != NoEval && !m.IsPromo() {
				gain := 0
				if m.IsEnPass() {
					gain = chess.SEEValues[chess.Pawn]
				} else if v := pos.PieceOn(m.To()); v != chess.NoPiece {
					gain = chess.SEEValues[v.Type()]
				}
				if rawEval+gain+150 <= alpha {
					continue
				}
			}
		}

		info := eval.MoveInfoFor(pos, m)
		pos.Make(m)
		if t.net != nil {
			t.net.Update(&t.accs[ply], &t.accs[ply+1], info)
		}
		t.ss[ply].currentMove = m
		t.ss[ply].contIdx = pieceTo(info.Piece, m.To())
		score := -t.qsearch(-beta, -alpha, ply+1)
		pos.Unmake(m)

		if t.s.stop.Load() {
			return 0
		}

		if score > bestScore {
			bestScore = score
			if score > alpha {
				bestMove = m
				if score >= beta {
					break
				}
				alpha = score
			}
		}
	}

	bound := BoundUpper
	if bestScore >= beta {
		bound = BoundLower
	}
	t.s.tt.Store(pos.Key(), bestMove, scoreToTT(bestScore, ply), rawEval, 0, bound)
	return bestScore
}
