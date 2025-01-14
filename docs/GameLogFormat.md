# Game Log Format

The self-play process produces game-log files, located in:

```
/workspace/output/<game>/<tag>/self-play-data/
```

Currently each file contains one game log. The log is recorded in a binary format, with the following sections:

```
1. Header
2. Final game state
3. Game outcome
4. Sampled indices
5. Memory offsets
6. Records
```

Sections 1-3 are of fixed-size.

Sections 4 and 5 contain a variable number of fixed-size elements. The header specifies how many of each.

Finally, section 6 contains a variable number of variable-sized elements, one per nonterminal state of the game.
Because they are variable-sized, we require memory-offset information to have O(1) random-access to the $k$'th
element. This offset information lives in section 5.

Each section, except for the last, is padded with zeros so that it fits in a multiple of 16 bytes.

Below are details about each section.

## Header

Struct definition: `core::GameLogBase::Header` in `cpp/include/core/GameLog.hpp`.

The header includes counts which dictate the size of sections 4 and 5.

## Final game state

The final state of the game when the game ends.

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

## Memory offsets

Consists of an array of `int32_t` entries. The $k$'th entry specifies the memory-offset, in bytes, for the $k$'th
record in the **Records** section, relative to the start of the **Records** section.

## Records

Consists of $n$ records. Each record consists of 3 components, concatenated together:

1. A `core::GameLog<Game>::Record` object
2. The first $a$ bytes of a `core::GameLog<Game>::TensorData` object specifying the policy-target, if recorded.
3. The first $b$ bytes of a `core::GameLog<Game>::TensorData` object specifying the action-values-target, if recorded.

For the `TensorData` objects, they can be recorded in a sparse-format, which has a variable-length. The object has
sufficient size to represent the tensor in dense-format, but if the sparse-format is used, then only enough of the
object is stored on disk to capture the full representation.
