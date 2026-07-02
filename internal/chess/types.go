// Package chess implements the board representation, move generation and
// core rules of standard chess. Correctness is the top priority: move
// generation is fully legal (never produces a move that leaves the own king
// in check) and is validated by an extensive perft test suite.
package chess

import (
	"fmt"
	"math/bits"
)

// Color of a side.
type Color uint8

const (
	White Color = 0
	Black Color = 1
)

func (c Color) Other() Color { return c ^ 1 }

// PieceType without color.
type PieceType uint8

const (
	Pawn PieceType = iota
	Knight
	Bishop
	Rook
	Queen
	King
	NoPieceType
)

// Piece combines color and type: piece = color*6 + type. 0..5 white, 6..11 black.
type Piece uint8

const NoPiece Piece = 12

func MakePiece(c Color, pt PieceType) Piece { return Piece(uint8(c)*6 + uint8(pt)) }

func (p Piece) Color() Color    { return Color(p / 6) }
func (p Piece) Type() PieceType { return PieceType(p % 6) }

var pieceChars = [13]byte{'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', '.'}

func (p Piece) Char() byte { return pieceChars[p] }

// Square 0..63, a1=0, b1=1, ..., h8=63.
type Square int8

const NoSquare Square = -1

const (
	SqA1 Square = iota
	SqB1
	SqC1
	SqD1
	SqE1
	SqF1
	SqG1
	SqH1
)

const (
	SqA8 Square = 56
	SqB8 Square = 57
	SqC8 Square = 58
	SqD8 Square = 59
	SqE8 Square = 60
	SqF8 Square = 61
	SqG8 Square = 62
	SqH8 Square = 63
)

func MakeSquare(file, rank int) Square { return Square(rank*8 + file) }

func (s Square) File() int { return int(s) & 7 }
func (s Square) Rank() int { return int(s) >> 3 }
func (s Square) BB() Bitboard {
	return 1 << uint(s)
}

// Flip vertically (a1 <-> a8) for black-perspective NNUE features.
func (s Square) FlipV() Square { return s ^ 56 }

func (s Square) String() string {
	if s == NoSquare {
		return "-"
	}
	return string([]byte{byte('a' + s.File()), byte('1' + s.Rank())})
}

func SquareFromString(str string) (Square, error) {
	if len(str) != 2 || str[0] < 'a' || str[0] > 'h' || str[1] < '1' || str[1] > '8' {
		return NoSquare, fmt.Errorf("bad square %q", str)
	}
	return MakeSquare(int(str[0]-'a'), int(str[1]-'1')), nil
}

// Bitboard is a 64-bit set of squares, bit i = square i.
type Bitboard uint64

const (
	FileABB Bitboard = 0x0101010101010101
	FileBBB Bitboard = FileABB << 1
	FileGBB Bitboard = FileABB << 6
	FileHBB Bitboard = FileABB << 7

	Rank1BB Bitboard = 0xFF
	Rank2BB Bitboard = Rank1BB << 8
	Rank3BB Bitboard = Rank1BB << 16
	Rank4BB Bitboard = Rank1BB << 24
	Rank5BB Bitboard = Rank1BB << 32
	Rank6BB Bitboard = Rank1BB << 40
	Rank7BB Bitboard = Rank1BB << 48
	Rank8BB Bitboard = Rank1BB << 56
)

func (b Bitboard) Count() int          { return bits.OnesCount64(uint64(b)) }
func (b Bitboard) LSB() Square         { return Square(bits.TrailingZeros64(uint64(b))) }
func (b Bitboard) IsSet(s Square) bool { return b&s.BB() != 0 }
func (b Bitboard) More() bool          { return b&(b-1) != 0 } // more than one bit set

// PopLSB removes and returns the lowest set square. b must be non-zero.
func (b *Bitboard) PopLSB() Square {
	s := b.LSB()
	*b &= *b - 1
	return s
}

// Move is a 16-bit encoded move.
//
//	bits 0..5   : to square
//	bits 6..11  : from square
//	bits 12..13 : promotion piece - Knight (only for promotions)
//	bits 14..15 : move kind (0 normal, 1 promotion, 2 en passant, 3 castling)
//
// Castling is encoded as the king move (e1g1 etc.).
type Move uint16

type MoveKind uint16

const (
	NormalMove    MoveKind = 0
	PromotionMove MoveKind = 1 << 14
	EnPassantMove MoveKind = 2 << 14
	CastleMove    MoveKind = 3 << 14
)

const NullMove Move = 0

func NewMove(from, to Square) Move {
	return Move(uint16(from)<<6 | uint16(to))
}

func NewPromotionMove(from, to Square, promo PieceType) Move {
	return Move(uint16(PromotionMove) | uint16(promo-Knight)<<12 | uint16(from)<<6 | uint16(to))
}

func NewEnPassantMove(from, to Square) Move {
	return Move(uint16(EnPassantMove) | uint16(from)<<6 | uint16(to))
}

func NewCastleMove(from, to Square) Move {
	return Move(uint16(CastleMove) | uint16(from)<<6 | uint16(to))
}

func (m Move) To() Square     { return Square(m & 63) }
func (m Move) From() Square   { return Square((m >> 6) & 63) }
func (m Move) Kind() MoveKind { return MoveKind(m) & (3 << 14) }
func (m Move) IsPromo() bool  { return m.Kind() == PromotionMove }
func (m Move) IsCastle() bool { return m.Kind() == CastleMove }
func (m Move) IsEnPass() bool { return m.Kind() == EnPassantMove }
func (m Move) PromoType() PieceType {
	return PieceType((m>>12)&3) + Knight
}

var promoChars = [7]byte{0, 'n', 'b', 'r', 'q', 0, 0}

// String returns the UCI representation of the move (e2e4, e7e8q, 0000).
func (m Move) String() string {
	if m == NullMove {
		return "0000"
	}
	s := m.From().String() + m.To().String()
	if m.IsPromo() {
		s += string(promoChars[m.PromoType()])
	}
	return s
}

// CastlingRights bit flags.
const (
	WhiteOO     uint8 = 1
	WhiteOOO    uint8 = 2
	BlackOO     uint8 = 4
	BlackOOO    uint8 = 8
	AllCastling uint8 = 15
)

// MoveList is a fixed-capacity move buffer used to avoid heap allocation
// in the search and generation hot paths.
type MoveList struct {
	Moves [256]Move
	N     int
}

func (ml *MoveList) Add(m Move)    { ml.Moves[ml.N] = m; ml.N++ }
func (ml *MoveList) Clear()        { ml.N = 0 }
func (ml *MoveList) Slice() []Move { return ml.Moves[:ml.N] }

// Contains reports whether the list contains m.
func (ml *MoveList) Contains(m Move) bool {
	for i := 0; i < ml.N; i++ {
		if ml.Moves[i] == m {
			return true
		}
	}
	return false
}
