// Package search implements the alpha-beta search: iterative deepening PVS
// with aspiration windows, a lock-free shared transposition table, staged
// move ordering, standard pruning/reduction/extension heuristics, lazy-SMP
// multithreading, MultiPV and pondering support.
package search

import (
	"math/bits"
	"sync/atomic"

	"fable/internal/chess"
)

// Score constants.
const (
	MaxPly = 128

	Inf       = 32500
	Mate      = 32000
	MateBound = Mate - 2*MaxPly // scores above this encode mate distances
	DrawScore = 0

	NoEval = -Inf // sentinel for "no static eval" (in check)
)

// Bound types stored in the TT.
const (
	BoundNone  uint8 = 0
	BoundUpper uint8 = 1
	BoundLower uint8 = 2
	BoundExact uint8 = 3
)

// TT is a lock-free transposition table. Each entry is two uint64 words:
// word0 = key ^ data, word1 = data. A torn read/write makes the validation
// key check fail, so races between threads are harmless (Hyatt's lockless
// hashing). Entries are grouped in buckets of 4 for replacement decisions.
//
// data packing (64 bits):
//
//	bits  0..15  move
//	bits 16..31  score (int16)
//	bits 32..47  static eval (int16)
//	bits 48..55  depth (uint8)
//	bits 56..57  bound
//	bits 58..63  generation (6 bits)
type TT struct {
	words []atomic.Uint64 // 2 words per entry
	slots uint64          // number of entries (multiple of bucketSize)
	gen   uint8
}

const bucketSize = 4

// NewTT allocates a table of the given size in MiB (min 1).
func NewTT(mb int) *TT {
	t := &TT{}
	t.Resize(mb)
	return t
}

// Resize reallocates the table. Must not be called during an active search.
func (t *TT) Resize(mb int) {
	if mb < 1 {
		mb = 1
	}
	slots := uint64(mb) * 1024 * 1024 / 16
	slots -= slots % bucketSize
	t.slots = slots
	t.words = make([]atomic.Uint64, 2*slots)
	t.gen = 0
}

// Clear wipes the table (ucinewgame).
func (t *TT) Clear() {
	for i := range t.words {
		t.words[i].Store(0)
	}
	t.gen = 0
}

// NewSearch advances the generation counter (call once per "go").
func (t *TT) NewSearch() { t.gen = (t.gen + 1) & 63 }

func (t *TT) bucketBase(key uint64) uint64 {
	hi, _ := bits.Mul64(key, t.slots/bucketSize)
	return hi * bucketSize
}

func packTT(move chess.Move, score, ev int16, depth uint8, bound, gen uint8) uint64 {
	return uint64(move) |
		uint64(uint16(score))<<16 |
		uint64(uint16(ev))<<32 |
		uint64(depth)<<48 |
		uint64(bound&3)<<56 |
		uint64(gen&63)<<58
}

// TTEntry is an unpacked probe result.
type TTEntry struct {
	Move  chess.Move
	Score int
	Eval  int
	Depth int
	Bound uint8
}

// Probe looks up key. Returns (entry, true) on hit.
func (t *TT) Probe(key uint64) (TTEntry, bool) {
	base := t.bucketBase(key)
	for i := uint64(0); i < bucketSize; i++ {
		w0 := t.words[2*(base+i)].Load()
		w1 := t.words[2*(base+i)+1].Load()
		if w0^w1 == key && w1 != 0 {
			return TTEntry{
				Move:  chess.Move(w1 & 0xFFFF),
				Score: int(int16(w1 >> 16)),
				Eval:  int(int16(w1 >> 32)),
				Depth: int((w1 >> 48) & 0xFF),
				Bound: uint8(w1>>56) & 3,
			}, true
		}
	}
	return TTEntry{}, false
}

// Store writes an entry with a depth/age-preferred replacement policy.
func (t *TT) Store(key uint64, move chess.Move, score, ev int, depth int, bound uint8) {
	if depth < 0 {
		depth = 0
	} else if depth > 255 {
		depth = 255
	}
	base := t.bucketBase(key)

	var victim uint64
	victimScore := int(1) << 30
	for i := uint64(0); i < bucketSize; i++ {
		w0 := t.words[2*(base+i)].Load()
		w1 := t.words[2*(base+i)+1].Load()
		if w1 == 0 {
			victim = base + i
			break
		}
		if w0^w1 == key {
			// Same position: keep the existing move if we have none, and
			// don't replace deeper exact data with shallow bound data.
			if move == chess.NullMove {
				move = chess.Move(w1 & 0xFFFF)
			}
			if bound != BoundExact && int((w1>>48)&0xFF) > depth+3 {
				return
			}
			victim = base + i
			break
		}
		entGen := uint8(w1>>58) & 63
		relAge := int(t.gen-entGen) & 63
		s := int((w1>>48)&0xFF) - relAge*8
		if s < victimScore {
			victimScore = s
			victim = base + i
		}
	}

	data := packTT(move, int16(score), int16(ev), uint8(depth), bound, t.gen)
	t.words[2*victim].Store(key ^ data)
	t.words[2*victim+1].Store(data)
}

// Hashfull estimates table saturation in permille (UCI "hashfull").
func (t *TT) Hashfull() int {
	n := uint64(1000)
	if t.slots < n {
		n = t.slots
	}
	cnt := 0
	for i := uint64(0); i < n; i++ {
		w1 := t.words[2*i+1].Load()
		if w1 != 0 && uint8(w1>>58)&63 == t.gen&63 {
			cnt++
		}
	}
	if n == 0 {
		return 0
	}
	return cnt * 1000 / int(n)
}

// scoreToTT converts a search score to a TT-storable score, making mate
// scores relative to the current node instead of the root.
func scoreToTT(score, ply int) int {
	if score >= MateBound {
		return score + ply
	}
	if score <= -MateBound {
		return score - ply
	}
	return score
}

// scoreFromTT is the inverse of scoreToTT.
func scoreFromTT(score, ply int) int {
	if score >= MateBound {
		return score - ply
	}
	if score <= -MateBound {
		return score + ply
	}
	return score
}
