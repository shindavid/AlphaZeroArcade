# CLAUDE.md — Python Architecture

## Directory Structure

```
py/
├── alphazero/
│   ├── dashboard/         # Plotting utilities for the web dashboard
│   ├── logic/             # Core logic: training, rating, game-log reading, ...
│   ├── scripts/           # Entry-point scripts (run_local.py, etc.)
│   └── servers/
│       ├── gaming/        # Self-play, ratings, eval servers (wrap C++ binary)
│       └── loop_control/  # LoopController + its managers
├── games/
│   ├── game_spec.py       # Abstract GameSpec base class
│   ├── index.py           # Registry: maps game name → GameSpec
│   └── {game}/spec.py     # Concrete GameSpec per game
├── shared/                # Shared types: model, loss_term, training_params, ...
├── build.py               # CMake build driver
├── run_tests.py           # Test runner
└── unit_tests/            # Python unit tests
```

## The AlphaZero Loop

`run_local.py -g {game} -t {tag}` orchestrates everything by launching these subprocesses:

```
run_loop_controller.py    → LoopController (Python, TCP port 1111)
run_self_play_server.py   → SelfPlayServer (Python) → spawns C++ worker process
run_eval_vs_benchmark_server.py → EvalVsBenchmarkServer (Python) → spawns C++ worker
run_ratings_server.py     → RatingsServer (Python) → spawns C++ worker   [optional]
```

Optional servers (`--run-ratings-server`, etc.) share the same CUDA device by default;
`-C N` limits to N devices.

## LoopController

Located in `py/alphazero/servers/loop_control/loop_controller.py`.

Acts as the central hub. Internally composed of **managers** that don't interact with each other
directly — all cross-manager communication goes through `LoopController`:

| Manager | Responsibility |
|---|---|
| `TrainingManager` | Sample training window, train neural net, export .onnx |
| `SelfPlayManager` | Coordinate self-play rounds, receive training data |
| `RatingsManager` | Rate model gens against reference players |
| `EvalVsBenchmarkManager` | Evaluate model gens against a benchmark committee |
| `SelfEvalManager` | Intra-run self-evaluation (used before a benchmark exists) |
| `LogSyncer` | Sync logs from scratch → persistent storage |
| `OutputDirSyncer` | Sync output dirs from scratch → persistent storage |
| `ClientConnectionManager` | Accept TCP connections from servers/workers |
| `DatabaseConnectionManager` | Manage SQLite connection pools |

## Training Loop (per generation)

1. **Gen 0 self-play**: C++ binary runs with `--no-model` (random/uniform policy), writes
   `gen-0.data` (50k positions by default).
2. **Train gen 1**: `TrainingManager` samples a window of positions, trains one or more
   minibatches, exports `gen-1.onnx`.
3. **Request more self-play**: loop controller tells self-play server to resume, using the new
   model.
4. **Rate gen 1**: `EvalVsBenchmarkManager` runs `alpha0-C` vs reference players to compute Elo.
5. Repeat: data accumulates, window slides forward (KataGo-style), training/eval runs in parallel.

## Neural Network

### Model Config

Each game's `GameSpec.model_configs` dict maps config name → `ModelConfigGenerator`.
The `default` config is used unless `--model-cfg` is specified. Each config declares the
network architecture (stem, trunk, heads) and the weighted loss terms to optimize.
Loss terms are defined in `py/shared/loss_term.py`; `TrainingManager` uses `NetTrainer`
to minimize their weighted sum.

### Checkpoints vs ONNX Models

- `checkpoints/gen-N.pt` — PyTorch checkpoint (weights + optimizer state), kept for resuming
- `models/gen-N.onnx` — ONNX export loaded by C++ workers via TensorRT

## GameSpec

Abstract base: `py/games/game_spec.py`. Each game implements:

| Property/Method | Purpose |
|---|---|
| `name` | Game name string (matches C++ binary name, e.g. `"c4"`) |
| `model_configs` | Dict of `ModelConfigGenerator` subclasses |
| `training_params` | Default `TrainingParams` (overrides global defaults) |
| `reference_player_family` | Defines reference player type + strength range for Elo |
| `training_options` | Extra CLI flags passed to C++ binary during self-play |

Registered in `py/games/index.py`. To add a new game: create `py/games/{game}/spec.py` with a
`GameSpec` subclass and register it in `py/games/index.py`.

## Server / Worker Protocol

All gaming servers follow the same pattern:
1. Server process connects to LoopController, receives a `client_id`
2. LoopController sends tasks as JSON (e.g., "run self-play with this model for N games")
3. Server spawns a C++ binary as a subprocess, passing:
   - `--client-role {role}` (e.g., `self-play-worker`, `eval-vs-benchmark-worker`)
   - `--loop-controller-hostname localhost --loop-controller-port 1111`
   - `--output-base-dir /workspace/mount/...`
4. C++ worker connects independently to LoopController and does its work

## Disk Layout (Cloud vs Local)

`LoopController` detects whether `~/scratch` and `/workspace` are on separate filesystems:

- **Cloud** (RunPod / GCP): fast work in `~/scratch/runs/{game}/{tag}/`,
  synced to `/workspace/output/{game}/{tag}/` by `OutputDirSyncer`
- **Local dev**: everything directly in `/workspace/output/{game}/{tag}/`

Log files similarly: `~/scratch/logs/{game}/{tag}/` vs `/workspace/output/{game}/{tag}/logs/`.

The Cloud functionality has not been tested recently.

## Databases (SQLite)

All state is persisted in SQLite databases under `databases/`:

| Database | Content |
|---|---|
| `clients.db` | Connected client history |
| `self-play.db` | Generations, cumulative position counts |
| `training.db` | Training loss history per generation |
| `evaluation/{benchmark_tag}.db` | Elo results vs benchmark |
