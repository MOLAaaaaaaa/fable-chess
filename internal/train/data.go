// Package train implements self-play data generation and NNUE training,
// entirely from zero human knowledge: generation 0 plays with the material
// bootstrap eval, every later generation with the previous network.
package train

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"fable/internal/chess"
	"fable/internal/eval"
)

// Record is one training position in a fixed 32-byte layout (little endian):
//
//	occ    uint64   occupancy bitboard
//	pieces [16]byte nibbles in ascending-square order of occ bits:
//	                bits 0..2 piece type (0=pawn..5=king), bit 3 color
//	stm    byte     side to move (0 white, 1 black)
//	score  int16    search score, centipawns, white POV
//	result byte     game result, white POV: 0 black win, 1 draw, 2 white win
//	ply    uint16   game ply of the position
//	gameID uint16   game identifier (low bits): lets the trainer split
//	                train/validation by game, avoiding leakage between
//	                highly correlated positions of the same game
const RecordSize = 32

type Record struct {
	Occ    uint64
	Pieces [16]byte
	Stm    uint8
	Score  int16
	Result uint8
	Ply    uint16
	GameID uint16

	// FileIdx is not serialized: it records which input file the record was
	// loaded from, so (FileIdx, GameID) identifies a game across files.
	FileIdx uint16
}

func (r *Record) Marshal(buf []byte) {
	binary.LittleEndian.PutUint64(buf[0:], r.Occ)
	copy(buf[8:24], r.Pieces[:])
	buf[24] = r.Stm
	binary.LittleEndian.PutUint16(buf[25:], uint16(r.Score))
	buf[27] = r.Result
	binary.LittleEndian.PutUint16(buf[28:], r.Ply)
	binary.LittleEndian.PutUint16(buf[30:], r.GameID)
}

func UnmarshalRecord(buf []byte) Record {
	var r Record
	r.Occ = binary.LittleEndian.Uint64(buf[0:])
	copy(r.Pieces[:], buf[8:24])
	r.Stm = buf[24]
	r.Score = int16(binary.LittleEndian.Uint16(buf[25:]))
	r.Result = buf[27]
	r.Ply = binary.LittleEndian.Uint16(buf[28:])
	r.GameID = binary.LittleEndian.Uint16(buf[30:])
	return r
}

// RecordFromPosition encodes the board of pos with the given white-POV score.
func RecordFromPosition(pos *chess.Position, scoreWhite int, ply int) Record {
	var r Record
	r.Occ = uint64(pos.Occupied())
	i := 0
	for occ := pos.Occupied(); occ != 0; i++ {
		sq := occ.PopLSB()
		pc := pos.PieceOn(sq)
		nib := uint8(pc.Type()) | uint8(pc.Color())<<3
		if i%2 == 0 {
			r.Pieces[i/2] |= nib
		} else {
			r.Pieces[i/2] |= nib << 4
		}
	}
	r.Stm = uint8(pos.Side())
	if scoreWhite > 32000 {
		scoreWhite = 32000
	} else if scoreWhite < -32000 {
		scoreWhite = -32000
	}
	r.Score = int16(scoreWhite)
	r.Ply = uint16(ply)
	return r
}

// visit iterates the (square, piece) pairs of the record.
func (r *Record) visit(f func(sq chess.Square, pc chess.Piece)) {
	i := 0
	for occ := chess.Bitboard(r.Occ); occ != 0; i++ {
		sq := occ.PopLSB()
		nib := r.Pieces[i/2]
		if i%2 == 1 {
			nib >>= 4
		}
		pt := chess.PieceType(nib & 7)
		c := chess.Color((nib >> 3) & 1)
		f(sq, chess.MakePiece(c, pt))
	}
}

// Features appends the white- and black-perspective feature indices.
func (r *Record) Features(wf, bf []uint16) ([]uint16, []uint16) {
	r.visit(func(sq chess.Square, pc chess.Piece) {
		wf = append(wf, uint16(eval.FeatureIndex(chess.White, pc, sq)))
		bf = append(bf, uint16(eval.FeatureIndex(chess.Black, pc, sq)))
	})
	return wf, bf
}

// FEN reconstructs a FEN for the record's board (castling/ep cleared; they
// do not affect NNUE features). Used for debugging and cross-checks.
func (r *Record) FEN() string {
	board := make([]byte, 64)
	for i := range board {
		board[i] = 0
	}
	r.visit(func(sq chess.Square, pc chess.Piece) {
		board[sq] = pc.Char()
	})
	var sb strings.Builder
	for rank := 7; rank >= 0; rank-- {
		empty := 0
		for file := 0; file < 8; file++ {
			ch := board[rank*8+file]
			if ch == 0 {
				empty++
				continue
			}
			if empty > 0 {
				sb.WriteByte(byte('0' + empty))
				empty = 0
			}
			sb.WriteByte(ch)
		}
		if empty > 0 {
			sb.WriteByte(byte('0' + empty))
		}
		if rank > 0 {
			sb.WriteByte('/')
		}
	}
	if r.Stm == 0 {
		sb.WriteString(" w - - 0 1")
	} else {
		sb.WriteString(" b - - 0 1")
	}
	return sb.String()
}

// Hash is a cheap content hash used for deduplication.
func (r *Record) Hash() uint64 {
	h := r.Occ*0x9E3779B97F4A7C15 ^ uint64(r.Stm)<<63
	for i, b := range r.Pieces {
		h ^= uint64(b) << (uint(i%8) * 8)
		h *= 0x100000001B3
	}
	return h
}

// LoadRecords reads all records from the given glob patterns / paths.
func LoadRecords(patterns []string) ([]Record, error) {
	var files []string
	for _, p := range patterns {
		matches, err := filepath.Glob(p)
		if err != nil {
			return nil, fmt.Errorf("bad pattern %q: %w", p, err)
		}
		if len(matches) == 0 {
			return nil, fmt.Errorf("no files match %q", p)
		}
		files = append(files, matches...)
	}
	var recs []Record
	for fi, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		if len(data)%RecordSize != 0 {
			return nil, fmt.Errorf("%s: size %d not a multiple of %d", f, len(data), RecordSize)
		}
		for off := 0; off < len(data); off += RecordSize {
			r := UnmarshalRecord(data[off : off+RecordSize])
			r.FileIdx = uint16(fi)
			recs = append(recs, r)
		}
	}
	return recs, nil
}

// recordWriter is a buffered append-only record sink.
type recordWriter struct {
	f   *os.File
	buf []byte
}

func newRecordWriter(path string, appendMode bool) (*recordWriter, error) {
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	flags := os.O_CREATE | os.O_WRONLY
	if appendMode {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	f, err := os.OpenFile(path, flags, 0o644)
	if err != nil {
		return nil, err
	}
	return &recordWriter{f: f}, nil
}

func (w *recordWriter) writeAll(recs []Record) error {
	var buf [RecordSize]byte
	for i := range recs {
		recs[i].Marshal(buf[:])
		w.buf = append(w.buf, buf[:]...)
	}
	if len(w.buf) >= 1<<20 {
		return w.flush()
	}
	return nil
}

func (w *recordWriter) flush() error {
	if len(w.buf) == 0 {
		return nil
	}
	_, err := w.f.Write(w.buf)
	w.buf = w.buf[:0]
	return err
}

func (w *recordWriter) close() error {
	if err := w.flush(); err != nil {
		w.f.Close()
		return err
	}
	return w.f.Close()
}

var _ io.Writer = (*os.File)(nil)
