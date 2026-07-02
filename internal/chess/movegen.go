package chess

// Fully legal move generation. Every move added to the list is guaranteed
// legal; there is no separate legality filter. Strategy:
//
//   - King moves are validated against enemy attacks computed with the king
//     removed from the occupancy (so sliders "see through" the king).
//   - In double check only king moves are generated.
//   - In single check non-king moves are restricted to capturing the checker
//     or blocking the check ray.
//   - Pinned pieces are restricted to the pin line.
//   - En passant legality (including all discovered-check edge cases) is
//     verified by direct simulation in epLegal.
//   - Castling paths are checked for occupancy and enemy attacks.

// pinnedBB returns our pieces that are absolutely pinned to our king.
func (p *Position) pinnedBB(us Color) Bitboard {
	ksq := p.KingSq(us)
	them := us.Other()
	snipers := (RookAttacks(ksq, 0) & (p.PiecesOf(them, Rook) | p.PiecesOf(them, Queen))) |
		(BishopAttacks(ksq, 0) & (p.PiecesOf(them, Bishop) | p.PiecesOf(them, Queen)))
	occ := p.Occupied()
	var pinned Bitboard
	for snipers != 0 {
		sn := snipers.PopLSB()
		between := BetweenBB[ksq][sn] & occ
		if between != 0 && !between.More() && between&p.byColor[us] != 0 {
			pinned |= between
		}
	}
	return pinned
}

func addPromotions(ml *MoveList, from, to Square) {
	ml.Add(NewPromotionMove(from, to, Queen))
	ml.Add(NewPromotionMove(from, to, Knight))
	ml.Add(NewPromotionMove(from, to, Rook))
	ml.Add(NewPromotionMove(from, to, Bishop))
}

// GenLegal fills ml with all legal moves. If capturesOnly is true, only
// captures and promotions are generated — except when in check, where all
// evasions are generated regardless (needed for correct quiescence search).
func (p *Position) GenLegal(ml *MoveList, capturesOnly bool) {
	ml.Clear()
	us := p.side
	them := us.Other()
	occ := p.Occupied()
	ourKing := p.KingSq(us)
	checkers := p.checkers
	inCheck := checkers != 0
	ourPieces := p.byColor[us]
	theirPieces := p.byColor[them]

	if inCheck {
		capturesOnly = false
	}

	// King moves.
	kTargets := KingAttacks[ourKing] &^ ourPieces
	if capturesOnly {
		kTargets &= theirPieces
	}
	occNoKing := occ &^ ourKing.BB()
	for b := kTargets; b != 0; {
		to := b.PopLSB()
		if p.AttackersTo(to, occNoKing)&theirPieces == 0 {
			ml.Add(NewMove(ourKing, to))
		}
	}

	if checkers.More() {
		return // double check: only king moves are legal
	}

	// Target squares for non-king moves.
	var targets Bitboard
	switch {
	case inCheck:
		targets = BetweenBB[ourKing][checkers.LSB()] | checkers
	case capturesOnly:
		targets = theirPieces
	default:
		targets = ^ourPieces
	}

	pinned := p.pinnedBB(us)

	// Knights: a pinned knight can never move.
	for b := p.PiecesOf(us, Knight) &^ pinned; b != 0; {
		from := b.PopLSB()
		for t := KnightAttacks[from] & targets; t != 0; {
			ml.Add(NewMove(from, t.PopLSB()))
		}
	}
	// Bishops.
	for b := p.PiecesOf(us, Bishop); b != 0; {
		from := b.PopLSB()
		t := BishopAttacks(from, occ) & targets
		if pinned.IsSet(from) {
			t &= LineBB[ourKing][from]
		}
		for t != 0 {
			ml.Add(NewMove(from, t.PopLSB()))
		}
	}
	// Rooks.
	for b := p.PiecesOf(us, Rook); b != 0; {
		from := b.PopLSB()
		t := RookAttacks(from, occ) & targets
		if pinned.IsSet(from) {
			t &= LineBB[ourKing][from]
		}
		for t != 0 {
			ml.Add(NewMove(from, t.PopLSB()))
		}
	}
	// Queens.
	for b := p.PiecesOf(us, Queen); b != 0; {
		from := b.PopLSB()
		t := QueenAttacks(from, occ) & targets
		if pinned.IsSet(from) {
			t &= LineBB[ourKing][from]
		}
		for t != 0 {
			ml.Add(NewMove(from, t.PopLSB()))
		}
	}

	// Pawns.
	p.genPawnMoves(ml, targets, pinned, capturesOnly)

	// Castling.
	if !inCheck && !capturesOnly {
		p.genCastling(ml, occ)
	}
}

func (p *Position) genPawnMoves(ml *MoveList, targets, pinned Bitboard, capturesOnly bool) {
	us := p.side
	them := us.Other()
	occ := p.Occupied()
	ksq := p.KingSq(us)
	enemies := p.byColor[them]

	var up Square
	var startRank, promoRank Bitboard
	if us == White {
		up, startRank, promoRank = 8, Rank2BB, Rank7BB
	} else {
		up, startRank, promoRank = -8, Rank7BB, Rank2BB
	}

	for b := p.PiecesOf(us, Pawn); b != 0; {
		from := b.PopLSB()
		pinLine := Bitboard(0xFFFFFFFFFFFFFFFF)
		if pinned.IsSet(from) {
			pinLine = LineBB[ksq][from]
		}
		isPromo := from.BB()&promoRank != 0

		// Captures.
		for t := PawnAttacks[us][from] & enemies & targets & pinLine; t != 0; {
			to := t.PopLSB()
			if isPromo {
				addPromotions(ml, from, to)
			} else {
				ml.Add(NewMove(from, to))
			}
		}

		// Pushes. In captures-only mode only promotion pushes are kept.
		if capturesOnly && !isPromo {
			continue
		}
		one := from + up
		if !occ.IsSet(one) {
			if one.BB()&targets&pinLine != 0 {
				if isPromo {
					addPromotions(ml, from, one)
				} else {
					ml.Add(NewMove(from, one))
				}
			}
			if !capturesOnly && from.BB()&startRank != 0 {
				two := one + up
				if !occ.IsSet(two) && two.BB()&targets&pinLine != 0 {
					ml.Add(NewMove(from, two))
				}
			}
		}
	}

	// En passant: legality (pins, discovered checks, check evasion) verified
	// by full simulation, so no target/pin masks are applied here.
	if ep := p.ep; ep != NoSquare {
		for cands := PawnAttacks[them][ep] & p.PiecesOf(us, Pawn); cands != 0; {
			from := cands.PopLSB()
			if p.epLegal(us, from, ep) {
				ml.Add(NewEnPassantMove(from, ep))
			}
		}
	}
}

func (p *Position) genCastling(ml *MoveList, occ Bitboard) {
	us := p.side
	them := us.Other()
	if us == White {
		if p.castling&WhiteOO != 0 &&
			occ&(SqF1.BB()|SqG1.BB()) == 0 &&
			!p.attackedBy(SqF1, them, occ) && !p.attackedBy(SqG1, them, occ) {
			ml.Add(NewCastleMove(SqE1, SqG1))
		}
		if p.castling&WhiteOOO != 0 &&
			occ&(SqB1.BB()|SqC1.BB()|SqD1.BB()) == 0 &&
			!p.attackedBy(SqD1, them, occ) && !p.attackedBy(SqC1, them, occ) {
			ml.Add(NewCastleMove(SqE1, SqC1))
		}
	} else {
		if p.castling&BlackOO != 0 &&
			occ&(SqF8.BB()|SqG8.BB()) == 0 &&
			!p.attackedBy(SqF8, them, occ) && !p.attackedBy(SqG8, them, occ) {
			ml.Add(NewCastleMove(SqE8, SqG8))
		}
		if p.castling&BlackOOO != 0 &&
			occ&(SqB8.BB()|SqC8.BB()|SqD8.BB()) == 0 &&
			!p.attackedBy(SqD8, them, occ) && !p.attackedBy(SqC8, them, occ) {
			ml.Add(NewCastleMove(SqE8, SqC8))
		}
	}
}

// HasLegalMoves reports whether the side to move has at least one legal move.
func (p *Position) HasLegalMoves() bool {
	var ml MoveList
	p.GenLegal(&ml, false)
	return ml.N > 0
}

// ParseUCIMove parses a move in UCI coordinate notation against the current
// position, returning the matching legal move. Returns NullMove if the move
// is not legal in this position.
func (p *Position) ParseUCIMove(s string) Move {
	var ml MoveList
	p.GenLegal(&ml, false)
	for i := 0; i < ml.N; i++ {
		if ml.Moves[i].String() == s {
			return ml.Moves[i]
		}
	}
	return NullMove
}
