# Game Log Format

The self-play process produces game-log files, located in:

```
/workspace/output/<game>/<tag>/self-play-data/
```

Currently each file contains one game log. The log is recorded in a binary format, with the following sections:

```
1. Header
2. Game outcome
3. Sampled indices
4. Actions
5. Policy target indices
6. Base states
7. Dense policy target
8. Sparse policy target entries
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

## Sampled indices

Consists of an array of `int32_t` entries. These indices indicate the game positions that are sampled for
network training purposes.

The number of these indices is specified in the header. On average, the number of sampled indices will be
$p\cdot k$, where:

* $p$ is the percentage of positions that are evaluated in "full-search" mode (see KataGo paper section 3.1)
* $k$ is the length of the game

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

The decision of whether to record a policy target in dense or sparse format is made dynamically at time of write, based on
the number of bytes that a sparse encoding would require.

## Base states

Each game has a game-specific `State` class. This is a POD struct representing a game state snapshot. See: [GameConcept.md](GameConcept.md) for details.

This section contains an array of `State` objects, for all states encountered in the game, whether sampled or not, including the
terminal state of the game.

When constructing the neural network input for a given (state-index, sym-index) from a game log, the ffi library loads the log file into memory,
reads the header to determine section memory offsets, and does a `reinterpret_cast<State>()` to the appropriate location in this section
to recover a `State`. In games where previous state history is included in the neural network input, the prior states can be found
in the preceding block of memory (thus avoiding the need to redundantly record recent-history).

The ffi library API includes a bool `apply_symmetry` parameter. When set to the default value of `true`, this `reinterpret_cast`'ed `State`
is transformed via a randomly selected symmetry.

## Dense policy targets

This section contains policy targets that are recorded in dense format. This is simply as an `Eigen::TensorFixedSize`, which is equivalent in byte-representation
to a `float[kNumGlobalActions]`.

## Sparse policy target entries

Struct definition: `core::GameLogBase::sparse_policy_entry_t` in `cpp/include/core/GameLog.hpp`.

This section contains (float, offset) pairs. A contiguous block of them encodes a given sparse policy target.
