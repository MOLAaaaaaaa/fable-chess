// Package eval provides position evaluation: an efficiently updatable
// neural network (NNUE) with SIMD-accelerated inference, and a minimal
// material fallback used to bootstrap self-play training from zero
// knowledge when no network file is available.
//
// Architecture: (768 -> H)x2 -> 1 with SCReLU activation.
//
//	Input features (per perspective): 12 piece kinds x 64 squares, with the
//	black perspective vertically mirrored and "own pieces first".
//	Accumulators are updated incrementally on make-move.
//	output = sum_i screlu(acc_stm[i]) * w[i] + sum_i screlu(acc_nstm[i]) * w[H+i] + bias
//
// Quantization: feature transformer weights/biases are int16 scaled by
// QA=255, output weights int16 scaled by QB=64, output bias int32 scaled by
// QA*QA*QB. Final centipawn score = raw * Scale / (QA*QA*QB).
package eval

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"

	"fable/internal/chess"
)

const (
	QA    = 255
	QB    = 64
	Scale = 400

	// NumFeatures is 12 piece kinds x 64 squares per perspective.
	NumFeatures = 768

	// MaxHidden bounds the accumulator arrays; hidden size must be a
	// multiple of 32 (SIMD kernel width) and <= MaxHidden.
	MaxHidden = 512

	// EvalMax bounds any static evaluation, well below mate scores.
	EvalMax = 25000
)

var netMagic = [8]byte{'F', 'A', 'B', 'L', 'E', 'N', 'N', '1'}

// Network is a quantized NNUE ready for inference.
type Network struct {
	Hidden     int
	FTWeights  []int16 // [NumFeatures][Hidden], feature-major
	FTBias     []int16 // [Hidden]
	OutWeights []int16 // [2*Hidden]: stm half then nstm half
	OutBias    int32   // scaled by QA*QA*QB
}

// ValidHidden reports whether h is a supported hidden layer size.
func ValidHidden(h int) bool {
	return h >= 32 && h <= MaxHidden && h%32 == 0
}

// FeatureIndex maps (perspective, piece, square) to the input feature index.
func FeatureIndex(persp chess.Color, pc chess.Piece, sq chess.Square) int {
	if persp == chess.Black {
		sq = sq.FlipV()
	}
	rel := 0
	if pc.Color() != persp {
		rel = 1
	}
	return (rel*6+int(pc.Type()))*64 + int(sq)
}

func (n *Network) ftRow(feature int) []int16 {
	off := feature * n.Hidden
	return n.FTWeights[off : off+n.Hidden : off+n.Hidden]
}

// NewNetwork allocates an empty (all-zero) network of the given hidden size.
func NewNetwork(hidden int) (*Network, error) {
	if !ValidHidden(hidden) {
		return nil, fmt.Errorf("nnue: unsupported hidden size %d (must be multiple of 32, 32..%d)", hidden, MaxHidden)
	}
	return &Network{
		Hidden:     hidden,
		FTWeights:  make([]int16, NumFeatures*hidden),
		FTBias:     make([]int16, hidden),
		OutWeights: make([]int16, 2*hidden),
	}, nil
}

// Save writes the network to path in the FABLENN1 format, creating parent
// directories as needed.
func (n *Network) Save(path string) error {
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	var buf bytes.Buffer
	buf.Write(netMagic[:])
	w32 := func(v uint32) { _ = binary.Write(&buf, binary.LittleEndian, v) }
	w32(1) // version
	w32(uint32(n.Hidden))
	w32(QA)
	w32(QB)
	w32(Scale)
	_ = binary.Write(&buf, binary.LittleEndian, n.FTWeights)
	_ = binary.Write(&buf, binary.LittleEndian, n.FTBias)
	_ = binary.Write(&buf, binary.LittleEndian, n.OutWeights)
	_ = binary.Write(&buf, binary.LittleEndian, n.OutBias)
	crc := crc32.ChecksumIEEE(buf.Bytes())
	_ = binary.Write(&buf, binary.LittleEndian, crc)
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

// Load reads a network from path, validating format and checksum.
func Load(path string) (*Network, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 32 {
		return nil, fmt.Errorf("nnue %s: file too small", path)
	}
	if !bytes.Equal(data[:8], netMagic[:]) {
		return nil, fmt.Errorf("nnue %s: bad magic", path)
	}
	crcWant := binary.LittleEndian.Uint32(data[len(data)-4:])
	if crc32.ChecksumIEEE(data[:len(data)-4]) != crcWant {
		return nil, fmt.Errorf("nnue %s: checksum mismatch (corrupt file)", path)
	}
	r := bytes.NewReader(data[8 : len(data)-4])
	var version, hidden, qa, qb, scale uint32
	for _, p := range []*uint32{&version, &hidden, &qa, &qb, &scale} {
		if err := binary.Read(r, binary.LittleEndian, p); err != nil {
			return nil, fmt.Errorf("nnue %s: truncated header", path)
		}
	}
	if version != 1 {
		return nil, fmt.Errorf("nnue %s: unsupported version %d", path, version)
	}
	if qa != QA || qb != QB || scale != Scale {
		return nil, fmt.Errorf("nnue %s: quantization scheme mismatch", path)
	}
	n, err := NewNetwork(int(hidden))
	if err != nil {
		return nil, fmt.Errorf("nnue %s: %w", path, err)
	}
	if err := binary.Read(r, binary.LittleEndian, n.FTWeights); err != nil {
		return nil, fmt.Errorf("nnue %s: truncated weights", path)
	}
	if err := binary.Read(r, binary.LittleEndian, n.FTBias); err != nil {
		return nil, fmt.Errorf("nnue %s: truncated bias", path)
	}
	if err := binary.Read(r, binary.LittleEndian, n.OutWeights); err != nil {
		return nil, fmt.Errorf("nnue %s: truncated output weights", path)
	}
	if err := binary.Read(r, binary.LittleEndian, &n.OutBias); err != nil {
		return nil, fmt.Errorf("nnue %s: truncated output bias", path)
	}
	if r.Len() != 0 {
		return nil, fmt.Errorf("nnue %s: trailing data", path)
	}
	// Guard the SIMD int16 product against out-of-range output weights.
	for _, w := range n.OutWeights {
		if w < -127 || w > 127 {
			return nil, fmt.Errorf("nnue %s: output weight %d out of range [-127,127]", path, w)
		}
	}
	return n, nil
}

// Accumulator holds the feature-transformer activations for both
// perspectives. Only the first Hidden entries of each row are meaningful.
type Accumulator struct {
	vals [2][MaxHidden]int16
}

// Refresh recomputes the accumulator from scratch for the given position.
func (n *Network) Refresh(pos *chess.Position, acc *Accumulator) {
	h := n.Hidden
	for c := 0; c < 2; c++ {
		copy(acc.vals[c][:h], n.FTBias)
	}
	occ := pos.Occupied()
	for occ != 0 {
		sq := occ.PopLSB()
		pc := pos.PieceOn(sq)
		addVec(acc.vals[0][:h], n.ftRow(FeatureIndex(chess.White, pc, sq)))
		addVec(acc.vals[1][:h], n.ftRow(FeatureIndex(chess.Black, pc, sq)))
	}
}

// MoveInfo captures everything needed to update accumulators incrementally.
// It must be filled from the position BEFORE the move is made.
type MoveInfo struct {
	Move     chess.Move
	Piece    chess.Piece // moving piece (pawn for promotions)
	Captured chess.Piece // NoPiece if none; the pawn for en passant
	Mover    chess.Color
}

// MoveInfoFor extracts MoveInfo for m from pos (which must be the position
// before making m).
func MoveInfoFor(pos *chess.Position, m chess.Move) MoveInfo {
	info := MoveInfo{
		Move:     m,
		Piece:    pos.PieceOn(m.From()),
		Captured: chess.NoPiece,
		Mover:    pos.Side(),
	}
	if m.IsEnPass() {
		info.Captured = chess.MakePiece(pos.Side().Other(), chess.Pawn)
	} else if !m.IsCastle() {
		info.Captured = pos.PieceOn(m.To())
	}
	return info
}

// Update computes child = parent + move delta, for both perspectives.
func (n *Network) Update(parent, child *Accumulator, info MoveInfo) {
	h := n.Hidden
	m := info.Move
	from, to := m.From(), m.To()
	us := info.Mover

	final := info.Piece
	if m.IsPromo() {
		final = chess.MakePiece(us, m.PromoType())
	}

	for c := chess.White; c <= chess.Black; c++ {
		src := parent.vals[c][:h]
		dst := child.vals[c][:h]
		addRow := n.ftRow(FeatureIndex(c, final, to))
		subRow := n.ftRow(FeatureIndex(c, info.Piece, from))

		switch {
		case m.IsCastle():
			rank := from.Rank()
			var rf, rt chess.Square
			if to > from {
				rf, rt = chess.MakeSquare(7, rank), chess.MakeSquare(5, rank)
			} else {
				rf, rt = chess.MakeSquare(0, rank), chess.MakeSquare(3, rank)
			}
			rook := chess.MakePiece(us, chess.Rook)
			addAddSubSubVec(dst, src,
				addRow,
				n.ftRow(FeatureIndex(c, rook, rt)),
				subRow,
				n.ftRow(FeatureIndex(c, rook, rf)))

		case info.Captured != chess.NoPiece:
			capSq := to
			if m.IsEnPass() {
				capSq = chess.MakeSquare(to.File(), from.Rank())
			}
			addSubSubVec(dst, src,
				addRow,
				subRow,
				n.ftRow(FeatureIndex(c, info.Captured, capSq)))

		default:
			addSubVec(dst, src, addRow, subRow)
		}
	}
}

// Copy copies the meaningful part of an accumulator (used for null moves).
func (n *Network) Copy(dst, src *Accumulator) {
	h := n.Hidden
	copy(dst.vals[0][:h], src.vals[0][:h])
	copy(dst.vals[1][:h], src.vals[1][:h])
}

// Forward computes the raw network output (scaled by QA*QA*QB) for the given
// side to move.
func (n *Network) Forward(acc *Accumulator, stm chess.Color) int64 {
	h := n.Hidden
	us := acc.vals[stm][:h]
	them := acc.vals[stm.Other()][:h]
	return outputVec(us, them, n.OutWeights) + int64(n.OutBias)
}

// Evaluate returns the network score in centipawns from the side to move's
// perspective.
func (n *Network) Evaluate(pos *chess.Position, acc *Accumulator) int {
	raw := n.Forward(acc, pos.Side())
	return clampCP(raw * Scale / (QA * QA * QB))
}

func clampCP(cp int64) int {
	if cp > EvalMax {
		cp = EvalMax
	} else if cp < -EvalMax {
		cp = -EvalMax
	}
	return int(cp)
}

// EvalFeatures computes the quantized evaluation directly from perspective
// feature lists. Used by the trainer to verify that the quantized network
// reproduces the float model through the exact engine arithmetic.
func (n *Network) EvalFeatures(wfeat, bfeat []uint16, stm chess.Color) int {
	h := n.Hidden
	var acc Accumulator
	copy(acc.vals[0][:h], n.FTBias)
	copy(acc.vals[1][:h], n.FTBias)
	for _, f := range wfeat {
		addVec(acc.vals[0][:h], n.ftRow(int(f)))
	}
	for _, f := range bfeat {
		addVec(acc.vals[1][:h], n.ftRow(int(f)))
	}
	raw := n.Forward(&acc, stm)
	return clampCP(raw * Scale / (QA * QA * QB))
}
