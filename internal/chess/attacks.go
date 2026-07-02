package chess

import "math/bits"

// Attack tables. Leaper attacks are simple lookups; slider attacks use
// "fancy" magic bitboards. Magic numbers are generated at package init with
// a fixed-seed PRNG; table construction inherently verifies each magic
// (a destructive collision makes construction fail and another magic is
// tried), so the tables are guaranteed correct by construction.

var (
	PawnAttacks   [2][64]Bitboard // [color][square]
	KnightAttacks [64]Bitboard
	KingAttacks   [64]Bitboard

	// BetweenBB[a][b]: squares strictly between a and b if aligned, else 0.
	BetweenBB [64][64]Bitboard
	// LineBB[a][b]: full line through a and b (incl. both) if aligned, else 0.
	LineBB [64][64]Bitboard
)

type magicEntry struct {
	mask    Bitboard
	magic   uint64
	shift   uint8
	attacks []Bitboard
}

var (
	rookMagics   [64]magicEntry
	bishopMagics [64]magicEntry
)

// RookAttacks returns rook attacks from s given board occupancy occ.
func RookAttacks(s Square, occ Bitboard) Bitboard {
	e := &rookMagics[s]
	return e.attacks[(uint64(occ&e.mask)*e.magic)>>e.shift]
}

// BishopAttacks returns bishop attacks from s given board occupancy occ.
func BishopAttacks(s Square, occ Bitboard) Bitboard {
	e := &bishopMagics[s]
	return e.attacks[(uint64(occ&e.mask)*e.magic)>>e.shift]
}

// QueenAttacks returns queen attacks from s given occupancy occ.
func QueenAttacks(s Square, occ Bitboard) Bitboard {
	return RookAttacks(s, occ) | BishopAttacks(s, occ)
}

// AttacksBB returns attacks for a piece type (not pawn) from s.
func AttacksBB(pt PieceType, s Square, occ Bitboard) Bitboard {
	switch pt {
	case Knight:
		return KnightAttacks[s]
	case Bishop:
		return BishopAttacks(s, occ)
	case Rook:
		return RookAttacks(s, occ)
	case Queen:
		return QueenAttacks(s, occ)
	case King:
		return KingAttacks[s]
	}
	return 0
}

// splitmix64 is a small deterministic PRNG used for magics and zobrist keys.
type splitmix64 struct{ x uint64 }

func (r *splitmix64) next() uint64 {
	r.x += 0x9E3779B97F4A7C15
	z := r.x
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

func (r *splitmix64) sparse() uint64 { return r.next() & r.next() & r.next() }

// slidingAttackSlow computes slider attacks by ray walking. Used only for
// table construction and by tests as a reference implementation.
func slidingAttackSlow(deltas [4][2]int, s Square, occ Bitboard) Bitboard {
	var att Bitboard
	f0, r0 := s.File(), s.Rank()
	for _, d := range deltas {
		f, r := f0+d[0], r0+d[1]
		for f >= 0 && f < 8 && r >= 0 && r < 8 {
			sq := MakeSquare(f, r)
			att |= sq.BB()
			if occ.IsSet(sq) {
				break
			}
			f += d[0]
			r += d[1]
		}
	}
	return att
}

var rookDeltas = [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
var bishopDeltas = [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}

func initMagics(magics *[64]magicEntry, deltas [4][2]int, rng *splitmix64) {
	for s := Square(0); s < 64; s++ {
		// Relevant occupancy mask: rays excluding board-edge terminal squares.
		edges := ((Rank1BB | Rank8BB) &^ (Rank1BB << uint(8*s.Rank()))) |
			((FileABB | FileHBB) &^ (FileABB << uint(s.File())))
		mask := slidingAttackSlow(deltas, s, 0) &^ edges
		bitsN := mask.Count()
		shift := uint8(64 - bitsN)
		size := 1 << bitsN

		// Enumerate all subsets of mask (carry-rippler) with reference attacks.
		occs := make([]Bitboard, size)
		refs := make([]Bitboard, size)
		b := Bitboard(0)
		for i := 0; ; i++ {
			occs[i] = b
			refs[i] = slidingAttackSlow(deltas, s, b)
			b = (b - mask) & mask
			if b == 0 {
				break
			}
		}

		attacks := make([]Bitboard, size)
		epoch := make([]int32, size)
		var gen int32
		for {
			magic := rng.sparse()
			// Quick rejection: need high bits well distributed.
			if bits.OnesCount64((uint64(mask)*magic)>>56) < 6 {
				continue
			}
			gen++
			ok := true
			for i := 0; i < size; i++ {
				idx := (uint64(occs[i]&mask) * magic) >> shift
				if epoch[idx] < gen {
					epoch[idx] = gen
					attacks[idx] = refs[i]
				} else if attacks[idx] != refs[i] {
					ok = false
					break
				}
			}
			if ok {
				magics[s] = magicEntry{mask: mask, magic: magic, shift: shift, attacks: attacks}
				break
			}
		}
	}
}

func init() {
	// Leapers.
	for s := Square(0); s < 64; s++ {
		f, r := s.File(), s.Rank()
		set := func(bb *Bitboard, df, dr int) {
			nf, nr := f+df, r+dr
			if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
				*bb |= MakeSquare(nf, nr).BB()
			}
		}
		var kn, kg, pw, pb Bitboard
		for _, d := range [][2]int{{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}} {
			set(&kn, d[0], d[1])
		}
		for _, d := range [][2]int{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}} {
			set(&kg, d[0], d[1])
		}
		set(&pw, -1, 1)
		set(&pw, 1, 1)
		set(&pb, -1, -1)
		set(&pb, 1, -1)
		KnightAttacks[s] = kn
		KingAttacks[s] = kg
		PawnAttacks[White][s] = pw
		PawnAttacks[Black][s] = pb
	}

	rng := &splitmix64{x: 0x5EED_C0DE_1234_5678}
	initMagics(&rookMagics, rookDeltas, rng)
	initMagics(&bishopMagics, bishopDeltas, rng)

	// Between / Line tables (built from slider attacks).
	for a := Square(0); a < 64; a++ {
		for b := Square(0); b < 64; b++ {
			if a == b {
				continue
			}
			switch {
			case RookAttacks(a, 0).IsSet(b):
				BetweenBB[a][b] = RookAttacks(a, b.BB()) & RookAttacks(b, a.BB())
				LineBB[a][b] = (RookAttacks(a, 0) & RookAttacks(b, 0)) | a.BB() | b.BB()
			case BishopAttacks(a, 0).IsSet(b):
				BetweenBB[a][b] = BishopAttacks(a, b.BB()) & BishopAttacks(b, a.BB())
				LineBB[a][b] = (BishopAttacks(a, 0) & BishopAttacks(b, 0)) | a.BB() | b.BB()
			}
		}
	}
}
