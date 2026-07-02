package search

import "fable/internal/chess"

// movePicker yields legal moves in a strong ordering. For rule safety all
// legal moves are generated once up front (so a corrupted TT move can never
// inject an illegal move — the TT move is only used if it matches a
// generated legal move), then selection-sorted lazily.
//
// Ordering: TT move, good captures/promotions (SEE >= 0, by MVV + capture
// history), killers, countermove, quiets by combined history, bad captures.
const (
	scoreTT        = 1 << 30
	scoreGoodCap   = 1 << 28
	scoreKiller1   = (1 << 27) + 2
	scoreKiller2   = (1 << 27) + 1
	scoreCounter   = 1 << 27
	scoreBadCap    = -(1 << 28)
	scoreQuietBase = 0 // quiet history scores are within ±3*histMax
)

type movePicker struct {
	list   chess.MoveList
	scores [256]int32
	idx    int
}

func (mp *movePicker) init(t *thread, ttMove chess.Move, ply int, capturesOnly bool) {
	pos := t.pos
	pos.GenLegal(&mp.list, capturesOnly)
	mp.idx = 0

	var counter chess.Move
	if ply >= 1 {
		if idx := t.ss[ply-1].contIdx; idx >= 0 {
			counter = t.hist.counter[idx]
		}
	}
	k1, k2 := t.ss[ply].killers[0], t.ss[ply].killers[1]

	for i := 0; i < mp.list.N; i++ {
		m := mp.list.Moves[i]
		var s int32
		switch {
		case m == ttMove:
			s = scoreTT
		case pos.IsCapture(m) || m.IsPromo():
			victim := chess.Pawn
			if !m.IsEnPass() {
				if v := pos.PieceOn(m.To()); v != chess.NoPiece {
					victim = v.Type()
				} else {
					victim = chess.NoPieceType // pure promotion push
				}
			}
			mvv := 0
			if victim != chess.NoPieceType {
				mvv = chess.SEEValues[victim]
			}
			if m.IsPromo() {
				mvv += chess.SEEValues[m.PromoType()] - chess.SEEValues[chess.Pawn]
			}
			capHist := 0
			if victim != chess.NoPieceType {
				capHist = int(t.hist.capture[pos.PieceOn(m.From())][m.To()][victim])
			}
			if pos.SEEGE(m, -50) {
				s = scoreGoodCap + int32(mvv*16+capHist)
			} else {
				s = scoreBadCap + int32(mvv*16+capHist)
			}
		case m == k1:
			s = scoreKiller1
		case m == k2:
			s = scoreKiller2
		case m == counter:
			s = scoreCounter
		default:
			s = int32(t.quietScore(m, ply))
		}
		mp.scores[i] = s
	}
}

// next returns the next-best move, or NullMove when exhausted. When
// skipQuiets is set, moves with quiet-range scores are skipped (killers,
// countermoves and history-scored quiets), but captures are still returned.
func (mp *movePicker) next(skipQuiets bool) chess.Move {
	for {
		if mp.idx >= mp.list.N {
			return chess.NullMove
		}
		best := mp.idx
		for i := mp.idx + 1; i < mp.list.N; i++ {
			if mp.scores[i] > mp.scores[best] {
				best = i
			}
		}
		mp.list.Moves[mp.idx], mp.list.Moves[best] = mp.list.Moves[best], mp.list.Moves[mp.idx]
		mp.scores[mp.idx], mp.scores[best] = mp.scores[best], mp.scores[mp.idx]
		m := mp.list.Moves[mp.idx]
		s := mp.scores[mp.idx]
		mp.idx++
		if skipQuiets && s < scoreGoodCap && s > scoreBadCap+(1<<26) && s != scoreTT {
			continue
		}
		return m
	}
}
