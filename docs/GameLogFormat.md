# Game Log Format

The self-play process produces game-log files, located in:

```
$A0A_OUTPUT_DIR/<game>/<tag>/self-play-data/
```

Currently each file contains one game log. The log is recorded in a binary format, with the following sections:

```
1. Header
2. Game outcome
3. Sym sample indices
4. Non-sym sample indices
5. Actions
6. Policy target indices
7. Base states
8. Dense policy target
9. Sparse policy target entries
```

The first two sections are of fixed-size.

The remaining sections each contain a variable number of a section-specific fixed-size struct. The header contains the variable numbers,
allowing one to read the `k`-th element of any of those sections by just reading the header, computing an offset, and then
scanning to that offset.

Each section is padded with zeros so that it fits in a multiple of 16 bytes.

Below are details about each section.

## Header

Struct definition: `core::GameLogBase::Header` in `cpp/include/core/GameLog.hpp`.

The header includes counts which dictate the size of the other sections. 

## Game outcome

The game outcome, recorded as an `Eigen::Array` of fixed size, which is equivalent in byte-representation
to a `float[kNumPlayers]`. 

## Sym sample indices

Struct definition: `core::GameLogBase::sym_sample_index_t` in `cpp/include/core/GameLog.hpp`.

Consists of an array of `sym_sample_index_t` entries. Each `sym_sample_index_t` consists of two 32-bit ints: a state-index, and a sym-index. 
The state-index dictates which position of the game to use, and the sym-index dictates which symmetry to apply.
To uniformly sample from the set of symmetry-expanded positions in a game, you would uniformly randomly select one of these,
scanning to a specific position in the Base States section based on the state-index, and then applying a symmetry
to that position based on sym-index.

The number of these indices is specified in the header. On average, the number of sym sample indices will be
$p\cdot s \cdot k$, where:

* $p$ is the percentage of positions that are evaluated in "full-search" mode (see KataGo paper section 3.1)
* $s$ is the average number of symmetry-expansions per position
* $k$ is the length of the game

## Non-sym sample indices

Struct definition: `core::GameLogBase::non_sym_sample_index_t` in `cpp/include/core/GameLog.hpp`.

Like sym sample indices, but without symmetry expansion. Each `non_sym_sample_index_t` consists of one 32-bit int: a single state-index. 

## Actions

An array of the actions taken in the game. Each action is represented as a 32-bit int.
The number of actions is generally one less than the number of states recorded (the terminal state
does not have an associated action).

## Policy target indices

Struct definition: `core::GameLogBase::policy_target_index_t` in `cpp/include/core/GameLog.hpp`.

For each recorded non-terminal position of the game, there is a corresponding `policy_target_index_t`. This is a compact
struct detailing where to find the data needed to reconstruct the policy target. There are three possibilities:

- Policy target not recorded (i.e., no sampled position needs this target for a training target)
- Policy target recorded in _dense_ format (i.e., as a `float[N]` where `N` is the global number of possible actions in the game)
- Policy target recorded in _sparse_ format (i.e., as an array of (float, offset) pairs for the nonzero entries of the policy target)

The `policy_target_index_t` struct either provides an int offset into the dense policy target section, or an int pair (start, end)
into the sparse policy target entry section.

## Base states

Each game has a game-specific `BaseState` class. This is a POD struct representing a game state snapshot. See: [GameConcept.md](GameConcept.md) for details.

This section contains an array of `BaseState` objects, for all states encountered in the game, whether sampled or not, including the
terminal state of the game.

When constructing the neural network input for a given (state-index, sym-index) from a game log, the ffi library loads the log file into memory,
reads the header to determine section memory offsets, and does a `reinterpret_cast<BaseState>()` to the appropriate location in this section
to recover a `BaseState`. In games where previous state history is included in the neural network input, the prior states can be found
in the preceding block of memory (thus avoiding the need to redundantly record recent-history).

## Dense policy targets

This section contains policy targets that are recorded in dense format. This is simply as an `Eigen::TensorFixedSize`, which is equivalent in byte-representation
to a `float[kNumGlobalActions]`. 

## Sparse policy target entries

Struct definition: `core::GameLogBase::sparse_policy_entry_t` in `cpp/include/core/GameLog.hpp`.

This section contains (float, offset) pairs. A contiguous block of them encodes a given sparse policy target.
