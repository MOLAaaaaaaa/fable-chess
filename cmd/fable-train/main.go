// fable-train — self-play data generation and NNUE training for Fable.
//
// Subcommands:
//
//	selfplay  generate training data by engine self-play
//	train     train a network from generated data
//	inspect   show statistics of data files
//	eval      evaluate a FEN with a network
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"

	"fable/internal/chess"
	"fable/internal/eval"
	"fable/internal/train"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "selfplay":
		err = cmdSelfplay(os.Args[2:])
	case "train":
		err = cmdTrain(os.Args[2:])
	case "match":
		err = cmdMatch(os.Args[2:])
	case "inspect":
		err = cmdInspect(os.Args[2:])
	case "eval":
		err = cmdEval(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
  fable-train selfplay -games N -nodes N -threads N [-net FILE] -out FILE [options]
  fable-train train    -data GLOB[,GLOB...] -out FILE [options]
  fable-train match    -a FILE|none -b FILE|none -games N [options]
  fable-train inspect  -data GLOB[,GLOB...]
  fable-train eval     -net FILE -fen FEN

run a subcommand with -h for its full option list`)
}

func cmdMatch(args []string) error {
	cfg := train.DefaultMatchConfig()
	fs := flag.NewFlagSet("match", flag.ExitOnError)
	fs.StringVar(&cfg.NetA, "a", "", "network A ('' or 'none' = material)")
	fs.StringVar(&cfg.NetB, "b", "", "network B ('' or 'none' = material)")
	fs.IntVar(&cfg.Games, "games", cfg.Games, "number of games (rounded to pairs)")
	nodes := fs.Int64("nodes", cfg.Nodes, "soft node limit per move")
	fs.IntVar(&cfg.Threads, "threads", runtime.NumCPU(), "parallel games")
	seed := fs.Uint64("seed", 1, "random seed")
	fs.IntVar(&cfg.RandomPlies, "random-plies", cfg.RandomPlies, "random opening plies")
	fs.IntVar(&cfg.HashMB, "hash", cfg.HashMB, "hash MB per engine")
	_ = fs.Parse(args)
	cfg.Nodes = *nodes
	cfg.Seed = *seed
	return train.RunMatch(cfg)
}

func cmdSelfplay(args []string) error {
	cfg := train.DefaultSelfplayConfig()
	fs := flag.NewFlagSet("selfplay", flag.ExitOnError)
	fs.IntVar(&cfg.Games, "games", cfg.Games, "number of games to play")
	nodes := fs.Int64("nodes", cfg.Nodes, "soft node limit per move")
	fs.IntVar(&cfg.Threads, "threads", runtime.NumCPU(), "parallel games")
	fs.StringVar(&cfg.NetPath, "net", "", "network file ('' or 'none' = material bootstrap)")
	fs.StringVar(&cfg.Out, "out", "data/selfplay.bin", "output data file")
	fs.BoolVar(&cfg.Append, "append", false, "append to output instead of truncating")
	seed := fs.Uint64("seed", 1, "random seed")
	fs.IntVar(&cfg.RandomPlies, "random-plies", cfg.RandomPlies, "random opening plies (alternates +1)")
	fs.IntVar(&cfg.OpeningCutoff, "opening-cutoff", cfg.OpeningCutoff, "max |cp| of accepted openings")
	fs.IntVar(&cfg.AdjWinCp, "adj-win", cfg.AdjWinCp, "win adjudication threshold, cp")
	fs.IntVar(&cfg.MaxPly, "max-ply", cfg.MaxPly, "maximum game length in plies")
	fs.IntVar(&cfg.HashMB, "hash", cfg.HashMB, "hash MB per worker")
	_ = fs.Parse(args)
	cfg.Nodes = *nodes
	cfg.Seed = *seed
	return train.RunSelfplay(cfg)
}

func cmdTrain(args []string) error {
	cfg := train.DefaultTrainConfig()
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	data := fs.String("data", "data/selfplay.bin", "comma-separated data file globs")
	out := fs.String("out", "fable.nnue", "output network file")
	fs.IntVar(&cfg.Hidden, "hidden", cfg.Hidden, "hidden layer size (multiple of 32)")
	fs.IntVar(&cfg.Epochs, "epochs", cfg.Epochs, "max training epochs")
	fs.IntVar(&cfg.BatchSize, "batch", cfg.BatchSize, "batch size")
	fs.Float64Var(&cfg.LR, "lr", cfg.LR, "learning rate")
	fs.Float64Var(&cfg.Lambda, "lambda", cfg.Lambda, "eval-vs-result target blend (1=pure eval, 0=pure WDL)")
	fs.Float64Var(&cfg.WeightDecay, "wd", cfg.WeightDecay, "weight decay (AdamW)")
	fs.Float64Var(&cfg.ValFrac, "val", cfg.ValFrac, "validation split fraction")
	fs.IntVar(&cfg.Patience, "patience", cfg.Patience, "early stopping patience (0=off)")
	fs.Int64Var(&cfg.Seed, "seed", cfg.Seed, "random seed")
	fs.IntVar(&cfg.Threads, "threads", cfg.Threads, "training threads")
	fs.BoolVar(&cfg.Dedup, "dedup", cfg.Dedup, "drop duplicate positions")
	fs.BoolVar(&cfg.Mirror, "mirror", cfg.Mirror, "horizontal-mirror data augmentation")
	_ = fs.Parse(args)
	cfg.Data = strings.Split(*data, ",")
	cfg.Out = *out
	return train.Train(cfg)
}

func cmdInspect(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ExitOnError)
	data := fs.String("data", "data/selfplay.bin", "comma-separated data file globs")
	_ = fs.Parse(args)
	recs, err := train.LoadRecords(strings.Split(*data, ","))
	if err != nil {
		return err
	}
	var w, d, l int
	var scoreSum float64
	hist := make(map[int]int) // score buckets of 100cp
	seen := make(map[uint64]struct{}, len(recs))
	dupes := 0
	for i := range recs {
		switch recs[i].Result {
		case 2:
			w++
		case 1:
			d++
		default:
			l++
		}
		scoreSum += float64(recs[i].Score)
		b := int(recs[i].Score) / 100
		if b > 10 {
			b = 10
		} else if b < -10 {
			b = -10
		}
		hist[b]++
		h := recs[i].Hash()
		if _, ok := seen[h]; ok {
			dupes++
		} else {
			seen[h] = struct{}{}
		}
	}
	n := len(recs)
	fmt.Printf("positions: %d (%d duplicate)\n", n, dupes)
	if n == 0 {
		return nil
	}
	fmt.Printf("results (white POV): +%d =%d -%d (%.1f%% / %.1f%% / %.1f%%)\n",
		w, d, l, 100*float64(w)/float64(n), 100*float64(d)/float64(n), 100*float64(l)/float64(n))
	fmt.Printf("mean score: %.1f cp\n", scoreSum/float64(n))
	fmt.Println("score histogram (100cp buckets, white POV):")
	for b := -10; b <= 10; b++ {
		if hist[b] > 0 {
			fmt.Printf("  %5d..%5d: %d\n", b*100, b*100+99, hist[b])
		}
	}
	fmt.Println("sample FENs:")
	for i := 0; i < min(3, n); i++ {
		r := recs[i*(n/max(1, min(3, n)))]
		fmt.Printf("  %s  score %d result %d ply %d\n", r.FEN(), r.Score, r.Result, r.Ply)
	}
	return nil
}

func cmdEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	netPath := fs.String("net", "fable.nnue", "network file")
	fen := fs.String("fen", chess.StartFEN, "position FEN")
	_ = fs.Parse(args)
	net, err := eval.Load(*netPath)
	if err != nil {
		return err
	}
	pos, err := chess.NewPosition(*fen)
	if err != nil {
		return err
	}
	var acc eval.Accumulator
	net.Refresh(pos, &acc)
	fmt.Printf("%s\nnnue eval (stm): %d cp\n", pos, net.Evaluate(pos, &acc))
	return nil
}
