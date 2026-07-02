package chess

// Static Exchange Evaluation: swap algorithm with x-ray attack updates.

// SEEValues are simple piece values used only for exchange evaluation and
// move ordering (search heuristics, not evaluation knowledge).
var SEEValues = [7]int{100, 320, 330, 500, 900, 20000, 0}

// SEEGE reports whether the static exchange value of move m is >= threshold.
// Castling moves are always >= 0. En passant and promotions are handled
// approximately but conservatively.
func (p *Position) SEEGE(m Move, threshold int) bool {
	if m.IsCastle() {
		return threshold <= 0
	}

	from, to := m.From(), m.To()

	swap := 0
	if m.IsEnPass() {
		swap = SEEValues[Pawn]
	} else if p.board[to] != NoPiece {
		swap = SEEValues[p.board[to].Type()]
	}
	swap -= threshold
	if swap < 0 {
		return false
	}

	// If we can lose the moving piece and still be >= threshold, it's a pass.
	next := p.board[from].Type()
	if m.IsPromo() {
		next = m.PromoType()
	}
	swap = SEEValues[next] - swap
	if swap <= 0 {
		return true
	}

	occ := p.Occupied() &^ from.BB() &^ to.BB()
	if m.IsEnPass() {
		occ &^= MakeSquare(to.File(), from.Rank()).BB()
	}
	occ |= to.BB()

	stm := p.side
	attackers := p.AttackersTo(to, occ)
	res := 1

	bishops := p.byType[Bishop] | p.byType[Queen]
	rooks := p.byType[Rook] | p.byType[Queen]

	for {
		stm = stm.Other()
		attackers &= occ
		stmAtt := attackers & p.byColor[stm]
		if stmAtt == 0 {
			break
		}
		res ^= 1

		// Pick the least valuable attacker.
		var bb Bitboard
		var pt PieceType
		for pt = Pawn; pt <= King; pt++ {
			if bb = stmAtt & p.byType[pt]; bb != 0 {
				break
			}
		}
		if pt == King {
			// The king can only complete the exchange if the opponent has
			// no attackers left; otherwise the "capture" is illegal.
			if attackers&p.byColor[stm.Other()]&occ != 0 {
				res ^= 1
			}
			break
		}

		swap = SEEValues[pt] - swap
		if swap < res {
			break
		}

		occ ^= bb & -bb // remove the attacker from the board
		// X-rays: recompute slider attacks through the vacated square.
		if pt == Pawn || pt == Bishop || pt == Queen {
			attackers |= BishopAttacks(to, occ) & bishops
		}
		if pt == Rook || pt == Queen {
			attackers |= RookAttacks(to, occ) & rooks
		}
	}
	return res == 1
}
