# Fable

[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go)](https://go.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-amd64%20%7C%20arm64-lightgrey)](#build)

A UCI chess engine written from scratch in Go, with NNUE evaluation and a
fully self-contained self-play training pipeline (zero human chess
knowledge, no external data, no external dependencies).

## Highlights

- **Board**: bitboards, fancy magic bitboards for sliders (magics generated
  and verified at startup), fully legal move generation (pin masks, check
  evasions, en-passant discovered-check simulation), zobrist hashing with
  en-passant normalization.
- **Search**: iterative-deepening principal-variation search with aspiration
  windows; lock-free shared transposition table (Hyatt XOR validation);
  quiescence with SEE pruning; null-move, reverse-futility, late-move
  pruning/reductions, singular extensions; killer/counter/butterfly/
  continuation/capture history ordering; lazy-SMP parallel search.
- **NNUE**: (768→H)×2 perspective network with SCReLU activation,
  incrementally updated accumulators, int16 quantization, hand-written
  AVX2 assembly kernels (with bit-exact pure-Go fallback verified by tests).
- **UCI**: `Hash`, `Threads`, `MultiPV`, `Ponder` (full
  `go ponder`/`ponderhit`/`stop` semantics), `Move Overhead`, `EvalFile`,
  `Clear Hash`, plus `searchmoves`, `mate`, `nodes`, `movetime`, `infinite`.
- **Rule safety**: engine-side detection of threefold repetition, the
  fifty-move rule, insufficient material, stalemate; a final legality check
  on every `bestmove` before it is printed. Validated by a perft suite over
  the classic tricky positions and make/unmake consistency tests.

## Build

Requires Go ≥ 1.21. No external dependencies.

```
go build -o fable.exe       ./cmd/fable        # the engine
go build -o fable-train.exe ./cmd/fable-train  # selfplay + trainer
```

Run tests (perft, SIMD equivalence, gradient checks, search sanity):

```
go test ./...
```

## Using the engine

`fable` speaks UCI on stdin/stdout. Point any chess GUI / match runner at
the binary. On startup it looks for `fable.nnue` next to the working
directory or the executable; if absent it falls back to a material-only
evaluation (that fallback is also what generation-0 training data is made
with).

```
setoption name Hash value 256
setoption name Threads value 8
setoption name MultiPV value 3
setoption name EvalFile value nets/gen2.nnue
```

Extra console commands: `bench [depth]`, `perft <depth>`, `d` (print
board), `eval`.

## Training pipeline (from zero)

Everything is engine self-play; no books, no tablebases, no external game
data. The typical loop:

```
# 1. generation 0: data from the material bootstrap eval
fable-train selfplay -games 4000 -nodes 8000 -threads 10 -out data/gen0.bin -adj-win 2000

# 2. train the first net (game-wise validation split, early stopping)
fable-train train -data data/gen0.bin -out nets/gen1.nnue -hidden 128

# 3. verify it actually got stronger before adopting it
fable-train match -a nets/gen1.nnue -b none -games 400 -nodes 8000 -threads 10

# 4. next generation: selfplay with the new net, retrain, re-verify
fable-train selfplay -games 4000 -nodes 8000 -threads 10 -net nets/gen1.nnue -out data/gen1.bin -adj-win 2000
fable-train train -data data/gen0.bin,data/gen1.bin -out nets/gen2.nnue -hidden 128
fable-train match -a nets/gen2.nnue -b nets/gen1.nnue -games 400 -nodes 8000 -threads 10

# 5. ship the winner
copy nets\gen2.nnue fable.nnue
```

`fable-train inspect -data ...` prints result balance, score histograms and
duplicate counts; `fable-train eval -net ... -fen ...` scores a single
position.

### Data generation details

Games start from random balanced openings (8–9 random plies, positions
outside ±250 cp rejected), are played at a fixed node budget per move, and
are adjudicated (win: |score| ≥ 2000 cp for 4 consecutive plies; draw: |score| ≤ 8
for 12 plies after ply 90, plus the normal rule draws). Saved positions are
filtered: quiet best move, not in check, past the opening randomness, |score|
≤ 3000 cp. Each record stores board, side to move, white-POV search score,
final game result and a game id.

### Overfitting countermeasures (small-data regime)

The pipeline is designed for runs of only a few thousand games:

- **Game-wise train/validation split.** Positions within one game are highly
  correlated; splitting by position leaks trajectories into the validation
  set and hides overfitting. The split is done on whole games via the stored
  game id.
- **Early stopping** on validation loss (patience 6) plus checkpointing of
  the best epoch, so the saved net is the val-loss minimum, not the last
  epoch.
- **Horizontal mirror augmentation** doubles the data using the exact
  left-right symmetry of chess rules (validation stays unmirrored).
- **Deduplication** of transposed/repeated positions before splitting.
- **Blended target**: `lambda·sigmoid(search_score) + (1−lambda)·game_result`
  (default λ = 0.8). The game result regularizes label noise from shallow
  searches; pure-result targets are too noisy at this scale.
- **Modest capacity default** (`-hidden 128`, ~200k parameters) and AdamW
  weight decay; hidden size is configurable in multiples of 32 up to 512.
- **Quantization check** after training reports the cp error between the
  float model and the int16 network actually used by the engine.

## Architecture notes

NNUE: input features are (piece type × color × square) = 768 per
perspective, "own pieces first" with the black perspective vertically
mirrored — so one weight matrix serves both colors and color symmetry is
exact by construction. Output: `sum screlu(acc_stm)·w₁ + sum
screlu(acc_nstm)·w₂ + b`, scaled to centipawns by 400/(255·255·64).
The network file (`FABLENN1`) is CRC-checked and quantization-scheme-checked
at load; output weights are range-validated so the AVX2 int16 kernels
cannot overflow.

Search safety: mate scores are ply-adjusted through the TT; TT cutoffs are
disabled near the fifty-move horizon (`rule50 ≥ 90`) and never override the
draw detectors, which run before the TT probe in every node.

## Repository layout

```
cmd/fable          UCI engine binary
cmd/fable-train    selfplay / train / match / inspect / eval tool
internal/chess     board, movegen, perft, SEE, zobrist
internal/eval      NNUE inference (+ AVX2 kernels), material bootstrap
internal/search    PVS search, TT, ordering, time management, lazy SMP
internal/train     data format, selfplay, trainer (Adam, mirror, early stop), match
internal/uci       UCI protocol front end
```
