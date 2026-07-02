// Fable — a UCI chess engine with NNUE evaluation.
//
// Usage:
//
//	fable            start in UCI mode
//	fable bench [d]  run the benchmark suite to depth d (default 12)
//	fable perft <d> [fen...]  run perft from a position (default startpos)
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"fable/internal/chess"
	"fable/internal/search"
	"fable/internal/uci"
)

func main() {
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "bench":
			depth := 12
			if len(os.Args) > 2 {
				if d, err := strconv.Atoi(os.Args[2]); err == nil {
					depth = d
				}
			}
			search.Bench(depth, os.Stdout)
			return
		case "perft":
			depth := 5
			if len(os.Args) > 2 {
				if d, err := strconv.Atoi(os.Args[2]); err == nil {
					depth = d
				}
			}
			fen := chess.StartFEN
			if len(os.Args) > 3 {
				fen = strings.Join(os.Args[3:], " ")
			}
			pos, err := chess.NewPosition(fen)
			if err != nil {
				fmt.Fprintln(os.Stderr, "bad fen:", err)
				os.Exit(1)
			}
			start := time.Now()
			div, total := chess.Divide(pos, depth)
			el := time.Since(start)
			fmt.Print(div)
			fmt.Printf("time %v (%.1f Mnps)\n", el, float64(total)/1e6/el.Seconds())
			return
		}
	}

	out := bufio.NewWriter(os.Stdout)
	uci.Run(os.Stdin, out)
	_ = out.Flush()
}
