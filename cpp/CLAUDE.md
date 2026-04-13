# CLAUDE.md — C++ Architecture

## Code

Make sure you clang-format each file that you edit. There is a convenience script
`py/tools/clang_format_all_cpp_files.py` to clang-format all c++ files in one go.

Multiline function implementations do not belong in header files. Headers should contain only
declarations. Implementations go in the corresponding `inline/**.inl` file. The header includes
the `.inl` at the bottom (after the closing namespace brace), using a path like:

```cpp
#include "inline/core/MyClass.inl"
```

Single-line function bodies (e.g. `bool is_win() const { return kind == kWin; }`) may remain in
the header.

This code-base is self-contained. No other project in the world uses it. As such, we have complete
freedom to change any component of the code in any way we wish to meet our objectives. If asked to
do a task, don't treat any of the code as untouchable, and instead seek out simple and natural
solutions. For example, if you need access to some private member of a class, just add a public
getter. If you need to invoke a helper function multiples times from inside a member function, don't
add it locally as an anonymous lambda, just add a helper method. Etc.

## Directory Structure

```
cpp/
├── include/                  # Headers (.hpp)
│   ├── alpha0/               # Alpha-zero search layer
│   ├── beta0/                # Beta-zero search layer
│   ├── core/                 # Core types: Game, PlayerFactory, ...
│   ├── games/{game}/         # Per-game: Game.hpp, Bindings.hpp, PlayerFactory.hpp
│   ├── generic_players/      # Unified player generators
│   ├── search/               # FfiMacro, search algorithm, NNEvaluation, ...
│   └── util/                 # MetaProgramming, CppUtil, Exceptions, ...
├── inline/                   # .inl files (template implementations)
├── src/                      # .cpp implementations + unit tests + goldenfile tests
│   ├── games/{game}/shared/  # {game}_ffi.cpp — FFI entry point per game
│   └── goldenfile_tests/     # Alpha0, GameServer, GenericPlayers unit tests
└── ...
```

## CMake Build: `add_game()` Macro

Defined in `cmake/GameMacros.cmake`. Calling `add_game(c4 ...)` produces four targets:

| CMake Target | Artifact | Purpose |
|---|---|---|
| `c4_common_objs` | object library | shared objects for all c4 targets |
| `c4_tests` | `bin/tests/c4_tests` | game-specific unit tests |
| `c4_exe` | `bin/c4` | standalone game binary |
| `c4_ffi` | `lib/libc4.so` | shared library loaded by Python (FFI) |

`targets.json` is written by CMake and read by `py/build.py` to know what targets exist
and what to build.

## Per-Game File Pattern

Each game has several key headers:

```
cpp/include/games/{game}/
├── Game.hpp           # Game type, state, actions — pure C++
├── Bindings.hpp       # Specializations + Bindings struct
├── PlayerFactory.hpp  # Registers all player types via mp::for_each
└── players/           # Game-specific player generators (HumanTui, Perfect, etc.)

cpp/src/games/{game}/shared/{game}_ffi.cpp
```

### `Bindings.hpp`

- Defines `struct Bindings { using SupportedSpecs = mp::TypeList<...>; }` listing all supported
  `Spec` instantiations. Each `Spec` is associated with a search-paradigm, like alpha0.

### `PlayerFactory.hpp`

Specifies what players exist for the given game. Does the work of parsing a command line
`--player "--type=..."` into a `PlayerGenerator`, which ultimately constructs the `Player`
instances.

### `{game}_ffi.cpp`

```cpp
#include "games/{game}/Bindings.hpp"
FFI_MACRO({ns}::Bindings)
```

`FFI_MACRO` expands to `extern "C"` functions that dispatch at runtime based on the paradigm
string (e.g., `"alpha0"`) using `mp::dispatch_type` over `Bindings::SupportedSpecs`. The python
side invokes these functions.

The purpose of the ffi library is to read game log files and produce input/target tensors.

## Binary Entry Point: `Main<PlayerFactory>`

`cpp/include/core/Main.hpp` / `cpp/inline/core/Main.inl`

`Main<PlayerFactory>` is the sole entry point template for every game binary. It constructs a
`GameServer` (or `GameServerProxy`) and runs a set of games between command-line specified players
in parallel. It also conditionally instantiates a `LoopControllerClient`, facilitating
communication with a python controller.

The game binary can either be called directly, or through the python server machinery.

## Game Servers: `GameServer` and `GameServerProxy`

### `GameServer<Game>` (`cpp/include/core/GameServer.hpp`)

The local game-running engine. Runs games in parallel using:
- **T threads** (`--num-game-threads`, default 16) — OS threads
- **P game slots** (`--parallelism`, default 1024) — independent game instances

T << P by design.

The code includes an important comment explaining the rationale for decoupling T and P.

When a `GameSlot` issues a yield (e.g., waiting for a NN evaluation batch
to complete), the thread moves on to the next available slot. This decouples CPU concurrency
from game concurrency and enables high GPU utilisation with few threads.

Note that this introduces considerable complexity to some player implementations, like `Manager`.

### `GameServerProxy<Game>` (`cpp/include/core/GameServerProxy.hpp`)

Two of the game processes can play against each other over a socket. This functionality was
motivated by the desire to save a given benchmark-run's binary/model to disk, and then to test
against it with future runs.

This inter-process playing is facilitated through `GameServerProxy` and `RemotePlayerProxy`. In
the main process, a `GameServer` is running, and the players of the remote process are registered
as `RemotePlayerProxy` instances. Those `RemotePlayerProxy` players simply forwards requests and
responses to the remote process, which is running a `GameServerProxy`, which has the actual
concrete player instance registered. The `GameServerProxy` in turn just forwards requests and
results from the main process and the concrete player.

## C++ Standard and Compiler

C++23 (`gnu++23`). Concepts, `if constexpr`, lambda templates (`[]<typename T>()`),
`std::type_identity` are all used freely.

## Adding a New Game

1. Create `cpp/include/games/{game}/` structure (see existing games)
2. Add `Bindings.hpp` with `Bindings::SupportedSpecs`
3. Add `PlayerFactory.hpp` using the `mp::for_each<Bindings::SupportedSpecs>` pattern
4. Add `cpp/src/games/{game}/shared/{game}_ffi.cpp` using `FFI_MACRO({ns}::Bindings)`
5. Add `add_game({game} ...)` to `CMakeLists.txt`
6. Add a `GameSpec` subclass in `py/games/{game}/spec.py` and register it in `py/games/index.py`

## Unit Tests

- Per-game tests: `cpp/src/games/{game}/test/` (built as `{game}_tests`)
- Library tests: `cpp/src/util/main/MetaProgrammingTests.cpp`, etc.
- Goldenfile tests: `cpp/src/goldenfile_tests/main/`: compare structured output against
  checked-in files in `goldenfiles/`. Update with `py/run_tests.py -w`.

## Interaction with python Loop Controller

When generating self-play data, the C++ binary is launched with `--client-role self-play-worker`
and communicates with the Python Loop Controller over TCP. The key message flow:

1. Loop Controller → Worker: `{"type": "pause"}`
   Worker pauses `TrainingDataWriter`, `NNEvaluationService`, `GameServer` (3-part receipt)
2. Loop Controller → Worker: `{"type": "reload-weights"}` — loads new `.onnx` model into TensorRT
3. Loop Controller → Worker: `{"type": "unpause"}` — resumes all threads
4. Loop Controller → Worker: `{"type": "data-pre-request", "n_rows_limit": N}`

TensorRT caches compiled engine plans in `/workspace/mount/TensorRT-cache/`.
