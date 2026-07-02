package search

import "fable/internal/chess"

// History heuristics, all per-thread (no cross-thread races):
//   - butterfly history: [side][from*64+to]
//   - capture history:   [piece][to][captured type]
//   - continuation history: [prev piece*64+prev to][piece*64+to], for the
//     previous move (countermove history) and the move two plies ago
//     (follow-up history)
//   - countermove table: [prev piece*64+prev to] -> move
const (
	histMax  = 16384
	pieceToN = 12 * 64
)

type history struct {
	butterfly [2][64 * 64]int16
	capture   [12][64][6]int16
	cont      [2][pieceToN][pieceToN]int16 // [0]: 1 ply ago, [1]: 2 plies ago
	counter   [pieceToN]chess.Move
}

func (h *history) clear() {
	*h = history{}
}

func histBonus(depth int) int {
	b := 300*depth - 300
	if b > 2400 {
		b = 2400
	}
	if b < 0 {
		b = 0
	}
	return b
}

func gravity(v *int16, bonus int) {
	b := bonus
	if b > histMax {
		b = histMax
	} else if b < -histMax {
		b = -histMax
	}
	*v += int16(b - int(*v)*abs(b)/histMax)
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func pieceTo(pc chess.Piece, to chess.Square) int {
	return int(pc)*64 + int(to)
}

// updateQuiet applies a bonus/malus to all quiet-move history tables.
func (t *thread) updateQuietHist(m chess.Move, bonus int, ply int) {
	pos := t.pos
	side := pos.Side()
	gravity(&t.hist.butterfly[side][int(m.From())*64+int(m.To())], bonus)
	pt := pieceTo(pos.PieceOn(m.From()), m.To())
	for i, back := range [2]int{1, 2} {
		if ply >= back {
			if idx := t.ss[ply-back].contIdx; idx >= 0 {
				gravity(&t.hist.cont[i][idx][pt], bonus)
			}
		}
	}
}

func (t *thread) updateCaptureHist(m chess.Move, bonus int) {
	pos := t.pos
	pc := pos.PieceOn(m.From())
	victim := chess.Pawn
	if !m.IsEnPass() {
		if v := pos.PieceOn(m.To()); v != chess.NoPiece {
			victim = v.Type()
		}
	}
	gravity(&t.hist.capture[pc][m.To()][victim], bonus)
}

// quietScore returns the combined ordering score for a quiet move.
func (t *thread) quietScore(m chess.Move, ply int) int {
	pos := t.pos
	side := pos.Side()
	s := int(t.hist.butterfly[side][int(m.From())*64+int(m.To())])
	pt := pieceTo(pos.PieceOn(m.From()), m.To())
	for i, back := range [2]int{1, 2} {
		if ply >= back {
			if idx := t.ss[ply-back].contIdx; idx >= 0 {
				s += int(t.hist.cont[i][idx][pt])
			}
		}
	}
	return s
}
