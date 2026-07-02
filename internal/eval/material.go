package eval

import "fable/internal/chess"

// Material is the zero-knowledge bootstrap evaluation used when no NNUE
// network is loaded (generation 0 of self-play training): material count
// plus a tiny generic mobility term and a tempo bonus. It deliberately
// avoids human-derived positional knowledge (no piece-square tables).
var matValues = [6]int{100, 300, 300, 500, 900, 0}

func Material(pos *chess.Position) int {
	score := 0
	for pt := chess.Pawn; pt <= chess.Queen; pt++ {
		score += matValues[pt] *
			(pos.PiecesOf(chess.White, pt).Count() - pos.PiecesOf(chess.Black, pt).Count())
	}

	occ := pos.Occupied()
	mob := 0
	for c := chess.White; c <= chess.Black; c++ {
		sign := 1
		if c == chess.Black {
			sign = -1
		}
		own := pos.ColorBB(c)
		for b := pos.PiecesOf(c, chess.Knight); b != 0; {
			mob += sign * (chess.KnightAttacks[b.PopLSB()] &^ own).Count()
		}
		for b := pos.PiecesOf(c, chess.Bishop); b != 0; {
			mob += sign * (chess.BishopAttacks(b.PopLSB(), occ) &^ own).Count()
		}
		for b := pos.PiecesOf(c, chess.Rook); b != 0; {
			mob += sign * (chess.RookAttacks(b.PopLSB(), occ) &^ own).Count()
		}
		for b := pos.PiecesOf(c, chess.Queen); b != 0; {
			mob += sign * (chess.QueenAttacks(b.PopLSB(), occ) &^ own).Count()
		}
	}
	score += 2 * mob

	if pos.Side() == chess.Black {
		score = -score
	}
	return score + 10 // tempo
}

// Evaluate is the static evaluation entry point used by the search:
// NNUE when a network is loaded, material bootstrap otherwise. The score is
// from the side to move's perspective and is damped as the 50-move counter
// grows so the engine steers away from drawn-out positions it thinks it is
// winning.
func Evaluate(pos *chess.Position, net *Network, acc *Accumulator) int {
	var v int
	if net != nil {
		v = net.Evaluate(pos, acc)
	} else {
		v = Material(pos)
	}
	r50 := pos.Rule50()
	if r50 > 100 {
		r50 = 100
	}
	v = v * (200 - r50) / 200
	if v > EvalMax {
		v = EvalMax
	} else if v < -EvalMax {
		v = -EvalMax
	}
	return v
}
