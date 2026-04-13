# CLAUDE.md — Project Overview

This is an **AlphaZero-style self-play learning framework** for multiple board games. The C++ side
handles game simulation, MCTS search, neural network inference (via TensorRT), and self-play data
generation. The Python side handles neural network training, system orchestration, and evaluation.

## Build

```bash
python py/build.py            # build everything (Release)
python py/build.py -d         # debug build → target/Debug/
python py/build.py -t c4      # build only c4 targets (by name or tag)
python py/build.py -t tests   # build only test binaries
python py/build.py -t c4,hex  # comma-separated targets
```

CMake output goes to `target/Release/` (or `target/Debug/`). Key outputs:
- `target/Release/bin/{game}` — game binary
- `target/Release/lib/lib{game}.so` — FFI shared library (loaded by Python)
- `target/Release/bin/tests/*_tests` — test binaries

The build reads `target/Release/targets.json` to know what was built.

## Run Tests

```bash
python py/run_tests.py              # all C++ tests + Python unit tests
python py/run_tests.py -c           # C++ tests only
python py/run_tests.py -p           # Python tests only
python py/run_tests.py -w           # write/update goldenfiles
```

C++ tests live in `target/Release/bin/tests/`. Python tests live in `py/unit_tests/`.
Goldenfile expected outputs live in `goldenfiles/`.

## Start an AlphaZero Training Run

```bash
python py/alphazero/scripts/run_local.py -g c4 -t MyTag
```

`-g` = game, `-t` = tag (arbitrary name, becomes part of output path).

This launches three subprocesses in parallel:
1. **Loop Controller** — trains the neural network, coordinates all activity
2. **Self-Play Server** — runs the C++ binary to play games using the current model
3. **Eval-vs-Benchmark Server** — evaluates each new model against reference players

Output is written to `/workspace/output/{game}/{tag}/` (see directory layout below).

## Output Directory Layout

```
output/{game}/{tag}/
├── bin/{game}          # copy of game binary used for this run
├── checkpoints/        # gen-N.pt (PyTorch checkpoints)
├── databases/
│   ├── clients.db
│   ├── self-play.db
│   ├── training.db
│   └── evaluation/
│       └── {benchmark_tag}.db
├── logs/
│   ├── loop-controller.log
│   ├── self-play-server/
│   ├── self-play-worker/
│   ├── eval-vs-benchmark-server/
│   └── eval-vs-benchmark-worker/
├── models/             # gen-N.onnx (TensorRT-compatible)
├── runtime/            # lock / freeze sentinel files
└── self-play-data/     # gen-N.data (raw training data)
```

On cloud setups (RunPod / GCP) with ephemeral local disks, active work uses `~/scratch/{...}` and
is synced to `/workspace/`. On local dev, everything uses `/workspace/` directly.

## Key Entry Points by Task

| Task | Script |
|------|--------|
| Start new run | `py/alphazero/scripts/run_local.py -g {game} -t {tag}` |
| View training dashboard | `py/alphazero/scripts/launch_dashboard.py -g {game}` |
| Fork and retrain | `py/alphazero/scripts/fork_run.py` + `run_local.py --train-only` |

## Architecture at a Glance

```
run_local.py
├── runs: run_loop_controller.py  (Python — trains NN, TCP port 1111)
├── runs: run_self_play_server.py (Python — spawns C++ binary as worker)
└── runs: run_eval_vs_benchmark_server.py (Python — spawns C++ binary as worker)

C++ binary (e.g. target/Release/bin/c4):
  --client-role self-play-worker    → runs games, writes gen-N.data
  --client-role eval-vs-benchmark-worker → plays MCTS vs reference player
```

The Loop Controller listens on TCP. All servers connect to it, then spawn C++ worker
subprocesses that communicate back independently.

## Search Paradigm

The search paradigm is AlphaZero MCTS (see `cpp/include/core/SearchParadigm.hpp`).

Player type strings used in C++ binary `--type` argument:
- `alpha0-T` — **Training** player (uses noisy policy for exploration)
- `alpha0-C` — **Competition** player (deterministic, for evaluation)

Each game's `Bindings::SupportedTraits` type-list declares the paradigm it supports.
