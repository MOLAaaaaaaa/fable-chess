package chess

import (
	"fmt"
	"sort"
	"strings"
)

// Perft counts leaf nodes of the legal move tree to the given depth,
// using bulk counting at depth 1.
func Perft(p *Position, depth int) uint64 {
	if depth <= 0 {
		return 1
	}
	var ml MoveList
	p.GenLegal(&ml, false)
	if depth == 1 {
		return uint64(ml.N)
	}
	var nodes uint64
	for i := 0; i < ml.N; i++ {
		p.Make(ml.Moves[i])
		nodes += Perft(p, depth-1)
		p.Unmake(ml.Moves[i])
	}
	return nodes
}

// Divide returns a per-root-move breakdown of perft counts, useful for
// debugging move generation against a reference engine.
func Divide(p *Position, depth int) (string, uint64) {
	var ml MoveList
	p.GenLegal(&ml, false)
	type entry struct {
		mv string
		n  uint64
	}
	entries := make([]entry, 0, ml.N)
	var total uint64
	for i := 0; i < ml.N; i++ {
		m := ml.Moves[i]
		p.Make(m)
		n := Perft(p, depth-1)
		p.Unmake(m)
		entries = append(entries, entry{m.String(), n})
		total += n
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].mv < entries[j].mv })
	var sb strings.Builder
	for _, e := range entries {
		fmt.Fprintf(&sb, "%s: %d\n", e.mv, e.n)
	}
	fmt.Fprintf(&sb, "total: %d\n", total)
	return sb.String(), total
}
