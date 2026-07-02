package chess

import (
	"fmt"
	"strconv"
	"strings"
)

// StartFEN is the standard chess starting position.
const StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

// Zobrist keys.
var (
	pieceKeys  [12][64]uint64
	sideKey    uint64
	castleKeys [16]uint64
	epKeys     [8]uint64
	castleMask [64]uint8
)

func init() {
	rng := &splitmix64{x: 0xA5B35705D5C0FFEE}
	for p := 0; p < 12; p++ {
		for s := 0; s < 64; s++ {
			pieceKeys[p][s] = rng.next()
		}
	}
	sideKey = rng.next()
	for i := range castleKeys {
		castleKeys[i] = rng.next()
	}
	for i := range epKeys {
		epKeys[i] = rng.next()
	}
	for i := range castleMask {
		castleMask[i] = AllCastling
	}
	castleMask[SqE1] &^= WhiteOO | WhiteOOO
	castleMask[SqH1] &^= WhiteOO
	castleMask[SqA1] &^= WhiteOOO
	castleMask[SqE8] &^= BlackOO | BlackOOO
	castleMask[SqH8] &^= BlackOO
	castleMask[SqA8] &^= BlackOOO
}

type stateInfo struct {
	key      uint64
	castling uint8
	ep       Square
	rule50   int16
	captured Piece
	checkers Bitboard
}

// Position holds a full chess position plus the undo stack and the zobrist
// key history used for repetition detection.
type Position struct {
	byType   [6]Bitboard
	byColor  [2]Bitboard
	board    [64]Piece
	side     Color
	castling uint8
	ep       Square
	rule50   int16
	fullmove int16
	key      uint64
	checkers Bitboard // pieces giving check to the side to move

	stack []stateInfo
	hist  []uint64 // zobrist keys of all ancestor positions (game + search path)

	// RootHist marks the boundary in hist between real game history and the
	// current search path. A repetition inside the search path counts as an
	// immediate draw; repetitions against game history require two prior
	// occurrences (true threefold).
	RootHist int
}

// Accessors.

func (p *Position) Side() Color                  { return p.side }
func (p *Position) Occupied() Bitboard           { return p.byColor[White] | p.byColor[Black] }
func (p *Position) ColorBB(c Color) Bitboard     { return p.byColor[c] }
func (p *Position) TypeBB(pt PieceType) Bitboard { return p.byType[pt] }
func (p *Position) PiecesOf(c Color, pt PieceType) Bitboard {
	return p.byColor[c] & p.byType[pt]
}
func (p *Position) PieceOn(s Square) Piece { return p.board[s] }
func (p *Position) KingSq(c Color) Square  { return (p.byColor[c] & p.byType[King]).LSB() }
func (p *Position) EPSquare() Square       { return p.ep }
func (p *Position) CastlingRights() uint8  { return p.castling }
func (p *Position) Key() uint64            { return p.key }
func (p *Position) Rule50() int            { return int(p.rule50) }
func (p *Position) FullMove() int          { return int(p.fullmove) }
func (p *Position) InCheck() bool          { return p.checkers != 0 }
func (p *Position) Checkers() Bitboard     { return p.checkers }
func (p *Position) HistLen() int           { return len(p.hist) }

// NonPawnMaterial reports whether c has any piece besides pawns and king.
func (p *Position) NonPawnMaterial(c Color) bool {
	return p.byColor[c]&(p.byType[Knight]|p.byType[Bishop]|p.byType[Rook]|p.byType[Queen]) != 0
}

// IsCapture reports whether m captures a piece (including en passant).
func (p *Position) IsCapture(m Move) bool {
	return (p.board[m.To()] != NoPiece && !m.IsCastle()) || m.IsEnPass()
}

// IsQuiet reports whether m is neither a capture nor a promotion.
func (p *Position) IsQuiet(m Move) bool {
	return !p.IsCapture(m) && !m.IsPromo()
}

func (p *Position) putPiece(pc Piece, s Square) {
	p.board[s] = pc
	b := s.BB()
	p.byType[pc.Type()] |= b
	p.byColor[pc.Color()] |= b
	p.key ^= pieceKeys[pc][s]
}

func (p *Position) removePiece(s Square) {
	pc := p.board[s]
	p.board[s] = NoPiece
	b := s.BB()
	p.byType[pc.Type()] &^= b
	p.byColor[pc.Color()] &^= b
	p.key ^= pieceKeys[pc][s]
}

func (p *Position) movePiece(from, to Square) {
	pc := p.board[from]
	ftb := from.BB() | to.BB()
	p.byType[pc.Type()] ^= ftb
	p.byColor[pc.Color()] ^= ftb
	p.board[from] = NoPiece
	p.board[to] = pc
	p.key ^= pieceKeys[pc][from] ^ pieceKeys[pc][to]
}

// AttackersTo returns all pieces (both colors) attacking square s, given
// occupancy occ.
func (p *Position) AttackersTo(s Square, occ Bitboard) Bitboard {
	return (PawnAttacks[Black][s] & p.PiecesOf(White, Pawn)) |
		(PawnAttacks[White][s] & p.PiecesOf(Black, Pawn)) |
		(KnightAttacks[s] & p.byType[Knight]) |
		(KingAttacks[s] & p.byType[King]) |
		(BishopAttacks(s, occ) & (p.byType[Bishop] | p.byType[Queen])) |
		(RookAttacks(s, occ) & (p.byType[Rook] | p.byType[Queen]))
}

func (p *Position) attackedBy(s Square, c Color, occ Bitboard) bool {
	return p.AttackersTo(s, occ)&p.byColor[c] != 0
}

func (p *Position) computeCheckers() Bitboard {
	us := p.side
	return p.AttackersTo(p.KingSq(us), p.Occupied()) & p.byColor[us.Other()]
}

func (p *Position) computeKey() uint64 {
	var k uint64
	for s := Square(0); s < 64; s++ {
		if pc := p.board[s]; pc != NoPiece {
			k ^= pieceKeys[pc][s]
		}
	}
	if p.side == Black {
		k ^= sideKey
	}
	k ^= castleKeys[p.castling]
	if p.ep != NoSquare {
		k ^= epKeys[p.ep.File()]
	}
	return k
}

// epLegal reports whether the pawn of color c on from can legally capture en
// passant to to (i.e. its own king is not left in check). Handles every edge
// case (pins, discovered checks through the captured pawn, resolving checks)
// by direct simulation.
func (p *Position) epLegal(c Color, from, to Square) bool {
	capSq := MakeSquare(to.File(), from.Rank())
	occ := (p.Occupied() &^ from.BB() &^ capSq.BB()) | to.BB()
	ksq := p.KingSq(c)
	them := c.Other()
	att := (PawnAttacks[c][ksq] & p.PiecesOf(them, Pawn) &^ capSq.BB()) |
		(KnightAttacks[ksq] & p.PiecesOf(them, Knight)) |
		(KingAttacks[ksq] & p.PiecesOf(them, King)) |
		(BishopAttacks(ksq, occ) & (p.PiecesOf(them, Bishop) | p.PiecesOf(them, Queen))) |
		(RookAttacks(ksq, occ) & (p.PiecesOf(them, Rook) | p.PiecesOf(them, Queen)))
	return att == 0
}

// epPossible reports whether mover has any legal en-passant capture to epSq.
// Used to normalize the ep square so that zobrist keys (and thus repetition
// detection) only differ when en passant is actually playable.
func (p *Position) epPossible(epSq Square, mover Color) bool {
	cands := PawnAttacks[mover.Other()][epSq] & p.PiecesOf(mover, Pawn)
	for cands != 0 {
		from := cands.PopLSB()
		if p.epLegal(mover, from, epSq) {
			return true
		}
	}
	return false
}

// Make plays a legal move m on the position.
func (p *Position) Make(m Move) {
	us := p.side
	them := us.Other()
	from, to := m.From(), m.To()
	pc := p.board[from]

	p.stack = append(p.stack, stateInfo{
		key: p.key, castling: p.castling, ep: p.ep,
		rule50: p.rule50, captured: NoPiece, checkers: p.checkers,
	})
	st := &p.stack[len(p.stack)-1]
	p.hist = append(p.hist, p.key)

	if p.ep != NoSquare {
		p.key ^= epKeys[p.ep.File()]
		p.ep = NoSquare
	}
	p.rule50++

	captured := NoPiece
	switch m.Kind() {
	case CastleMove:
		var rfrom, rto Square
		if to > from { // king side
			rfrom, rto = MakeSquare(7, int(from.Rank())), MakeSquare(5, int(from.Rank()))
		} else {
			rfrom, rto = MakeSquare(0, int(from.Rank())), MakeSquare(3, int(from.Rank()))
		}
		p.movePiece(from, to)
		p.movePiece(rfrom, rto)

	case EnPassantMove:
		capSq := MakeSquare(int(to.File()), int(from.Rank()))
		captured = p.board[capSq]
		p.removePiece(capSq)
		p.movePiece(from, to)
		p.rule50 = 0

	default: // normal move or promotion
		if p.board[to] != NoPiece {
			captured = p.board[to]
			p.removePiece(to)
			p.rule50 = 0
		}
		p.movePiece(from, to)
		if pc.Type() == Pawn {
			p.rule50 = 0
			if m.Kind() == PromotionMove {
				p.removePiece(to)
				p.putPiece(MakePiece(us, m.PromoType()), to)
			} else if int(to)-int(from) == 16 || int(from)-int(to) == 16 {
				epSq := Square((int(from) + int(to)) / 2)
				if p.epPossible(epSq, them) {
					p.ep = epSq
					p.key ^= epKeys[epSq.File()]
				}
			}
		}
	}
	st.captured = captured

	if newCr := p.castling & castleMask[from] & castleMask[to]; newCr != p.castling {
		p.key ^= castleKeys[p.castling] ^ castleKeys[newCr]
		p.castling = newCr
	}

	p.side = them
	p.key ^= sideKey
	if them == White {
		p.fullmove++
	}
	p.checkers = p.computeCheckers()
}

// Unmake takes back move m (must be the last move made).
func (p *Position) Unmake(m Move) {
	st := p.stack[len(p.stack)-1]
	p.stack = p.stack[:len(p.stack)-1]
	p.hist = p.hist[:len(p.hist)-1]

	us := p.side.Other() // the side that made the move
	from, to := m.From(), m.To()

	switch m.Kind() {
	case CastleMove:
		var rfrom, rto Square
		if to > from {
			rfrom, rto = MakeSquare(7, int(from.Rank())), MakeSquare(5, int(from.Rank()))
		} else {
			rfrom, rto = MakeSquare(0, int(from.Rank())), MakeSquare(3, int(from.Rank()))
		}
		p.movePiece(to, from)
		p.movePiece(rto, rfrom)

	case EnPassantMove:
		p.movePiece(to, from)
		p.putPiece(st.captured, MakeSquare(int(to.File()), int(from.Rank())))

	default:
		if m.Kind() == PromotionMove {
			p.removePiece(to)
			p.putPiece(MakePiece(us, Pawn), to)
		}
		p.movePiece(to, from)
		if st.captured != NoPiece {
			p.putPiece(st.captured, to)
		}
	}

	p.side = us
	if us == Black {
		p.fullmove--
	}
	p.key = st.key
	p.castling = st.castling
	p.ep = st.ep
	p.rule50 = st.rule50
	p.checkers = st.checkers
}

// MakeNull plays a null move (side to move passes). Only legal when not in check.
func (p *Position) MakeNull() {
	p.stack = append(p.stack, stateInfo{
		key: p.key, castling: p.castling, ep: p.ep,
		rule50: p.rule50, captured: NoPiece, checkers: p.checkers,
	})
	p.hist = append(p.hist, p.key)
	if p.ep != NoSquare {
		p.key ^= epKeys[p.ep.File()]
		p.ep = NoSquare
	}
	p.rule50++
	p.side = p.side.Other()
	p.key ^= sideKey
	p.checkers = 0
}

// UnmakeNull takes back a null move.
func (p *Position) UnmakeNull() {
	st := p.stack[len(p.stack)-1]
	p.stack = p.stack[:len(p.stack)-1]
	p.hist = p.hist[:len(p.hist)-1]
	p.side = p.side.Other()
	p.key = st.key
	p.castling = st.castling
	p.ep = st.ep
	p.rule50 = st.rule50
	p.checkers = st.checkers
}

// IsRepetition reports whether the current position is a repetition draw:
// either a twofold repetition within the current search path, or a true
// threefold repetition counting real game history.
func (p *Position) IsRepetition() bool {
	n := len(p.hist)
	count := 0
	for d := 2; d <= int(p.rule50) && n-d >= 0; d += 2 {
		if p.hist[n-d] == p.key {
			if n-d >= p.RootHist {
				return true
			}
			count++
			if count >= 2 {
				return true
			}
		}
	}
	return false
}

// InsufficientMaterial reports positions that are provably dead draws:
// KvK, KNvK, KBvK.
func (p *Position) InsufficientMaterial() bool {
	if p.byType[Pawn]|p.byType[Rook]|p.byType[Queen] != 0 {
		return false
	}
	return (p.byType[Knight] | p.byType[Bishop]).Count() <= 1
}

// Clone returns an independent deep copy (fresh stack/history slices).
func (p *Position) Clone() *Position {
	q := *p
	q.stack = make([]stateInfo, len(p.stack), len(p.stack)+256)
	copy(q.stack, p.stack)
	q.hist = make([]uint64, len(p.hist), len(p.hist)+256)
	copy(q.hist, p.hist)
	return &q
}

// NewPosition parses a FEN string, validating and sanitizing it.
func NewPosition(fen string) (*Position, error) {
	fields := strings.Fields(fen)
	if len(fields) < 2 {
		return nil, fmt.Errorf("fen: need at least board and side fields: %q", fen)
	}
	p := &Position{
		ep:       NoSquare,
		fullmove: 1,
		stack:    make([]stateInfo, 0, 256),
		hist:     make([]uint64, 0, 256),
	}
	for i := range p.board {
		p.board[i] = NoPiece
	}

	// Board.
	ranks := strings.Split(fields[0], "/")
	if len(ranks) != 8 {
		return nil, fmt.Errorf("fen: board must have 8 ranks: %q", fields[0])
	}
	for r := 0; r < 8; r++ {
		rank := 7 - r
		file := 0
		for _, ch := range []byte(ranks[r]) {
			if ch >= '1' && ch <= '8' {
				file += int(ch - '0')
				continue
			}
			if file > 7 {
				return nil, fmt.Errorf("fen: rank %d overflow", rank+1)
			}
			var pc Piece = NoPiece
			for i, c := range pieceChars[:12] {
				if c == ch {
					pc = Piece(i)
					break
				}
			}
			if pc == NoPiece {
				return nil, fmt.Errorf("fen: bad piece char %q", ch)
			}
			sq := MakeSquare(file, rank)
			p.board[sq] = pc
			p.byType[pc.Type()] |= sq.BB()
			p.byColor[pc.Color()] |= sq.BB()
			file++
		}
		if file != 8 {
			return nil, fmt.Errorf("fen: rank %d has %d files", rank+1, file)
		}
	}
	if p.PiecesOf(White, King).Count() != 1 || p.PiecesOf(Black, King).Count() != 1 {
		return nil, fmt.Errorf("fen: each side must have exactly one king")
	}
	if p.byType[Pawn]&(Rank1BB|Rank8BB) != 0 {
		return nil, fmt.Errorf("fen: pawns on back rank")
	}

	// Side to move.
	switch fields[1] {
	case "w":
		p.side = White
	case "b":
		p.side = Black
	default:
		return nil, fmt.Errorf("fen: bad side %q", fields[1])
	}

	// The side NOT to move must not be in check (otherwise the position is
	// illegal and move generation invariants break).
	if p.attackedBy(p.KingSq(p.side.Other()), p.side, p.Occupied()) {
		return nil, fmt.Errorf("fen: side not to move is in check")
	}

	// Castling rights, sanitized against actual piece placement.
	if len(fields) > 2 && fields[2] != "-" {
		for _, ch := range fields[2] {
			switch ch {
			case 'K':
				p.castling |= WhiteOO
			case 'Q':
				p.castling |= WhiteOOO
			case 'k':
				p.castling |= BlackOO
			case 'q':
				p.castling |= BlackOOO
			default:
				return nil, fmt.Errorf("fen: bad castling char %q", ch)
			}
		}
	}
	if p.board[SqE1] != MakePiece(White, King) {
		p.castling &^= WhiteOO | WhiteOOO
	}
	if p.board[SqH1] != MakePiece(White, Rook) {
		p.castling &^= WhiteOO
	}
	if p.board[SqA1] != MakePiece(White, Rook) {
		p.castling &^= WhiteOOO
	}
	if p.board[SqE8] != MakePiece(Black, King) {
		p.castling &^= BlackOO | BlackOOO
	}
	if p.board[SqH8] != MakePiece(Black, Rook) {
		p.castling &^= BlackOO
	}
	if p.board[SqA8] != MakePiece(Black, Rook) {
		p.castling &^= BlackOOO
	}

	// En passant square, normalized: kept only if a legal ep capture exists.
	if len(fields) > 3 && fields[3] != "-" {
		sq, err := SquareFromString(fields[3])
		if err != nil {
			return nil, fmt.Errorf("fen: bad ep square %q", fields[3])
		}
		wantRank := 5
		if p.side == Black {
			wantRank = 2
		}
		if sq.Rank() != wantRank {
			return nil, fmt.Errorf("fen: ep square %s on wrong rank", sq)
		}
		// The pawn that just double-pushed must be present and the square
		// behind it empty; otherwise ignore the ep field.
		victim := MakeSquare(sq.File(), 4)
		if p.side == Black {
			victim = MakeSquare(sq.File(), 3)
		}
		if p.board[victim] == MakePiece(p.side.Other(), Pawn) &&
			p.board[sq] == NoPiece && p.epPossible(sq, p.side) {
			p.ep = sq
		}
	}

	// Move counters.
	if len(fields) > 4 {
		v, err := strconv.Atoi(fields[4])
		if err != nil || v < 0 {
			return nil, fmt.Errorf("fen: bad halfmove clock %q", fields[4])
		}
		p.rule50 = int16(v)
	}
	if len(fields) > 5 {
		v, err := strconv.Atoi(fields[5])
		if err != nil || v < 1 {
			return nil, fmt.Errorf("fen: bad fullmove number %q", fields[5])
		}
		p.fullmove = int16(v)
	}

	p.key = p.computeKey()
	p.checkers = p.computeCheckers()
	return p, nil
}

// MustPosition parses a FEN and panics on error (for tests/known-good FENs).
func MustPosition(fen string) *Position {
	p, err := NewPosition(fen)
	if err != nil {
		panic(err)
	}
	return p
}

// FEN renders the position as a FEN string.
func (p *Position) FEN() string {
	var sb strings.Builder
	for rank := 7; rank >= 0; rank-- {
		empty := 0
		for file := 0; file < 8; file++ {
			pc := p.board[MakeSquare(file, rank)]
			if pc == NoPiece {
				empty++
				continue
			}
			if empty > 0 {
				sb.WriteByte(byte('0' + empty))
				empty = 0
			}
			sb.WriteByte(pc.Char())
		}
		if empty > 0 {
			sb.WriteByte(byte('0' + empty))
		}
		if rank > 0 {
			sb.WriteByte('/')
		}
	}
	sb.WriteByte(' ')
	if p.side == White {
		sb.WriteByte('w')
	} else {
		sb.WriteByte('b')
	}
	sb.WriteByte(' ')
	if p.castling == 0 {
		sb.WriteByte('-')
	} else {
		if p.castling&WhiteOO != 0 {
			sb.WriteByte('K')
		}
		if p.castling&WhiteOOO != 0 {
			sb.WriteByte('Q')
		}
		if p.castling&BlackOO != 0 {
			sb.WriteByte('k')
		}
		if p.castling&BlackOOO != 0 {
			sb.WriteByte('q')
		}
	}
	fmt.Fprintf(&sb, " %s %d %d", p.ep.String(), p.rule50, p.fullmove)
	return sb.String()
}

// String renders an ASCII board for debugging.
func (p *Position) String() string {
	var sb strings.Builder
	for rank := 7; rank >= 0; rank-- {
		fmt.Fprintf(&sb, "%d ", rank+1)
		for file := 0; file < 8; file++ {
			sb.WriteByte(p.board[MakeSquare(file, rank)].Char())
			sb.WriteByte(' ')
		}
		sb.WriteByte('\n')
	}
	sb.WriteString("  a b c d e f g h\n")
	sb.WriteString(p.FEN())
	return sb.String()
}
