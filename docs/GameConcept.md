# Game Concept

To add a new game, you need to supply a `Game` class that adheres to the `core::concepts::Game` concept, defined in `cpp/include/core/concepts/Game.hpp`.

For an example, see `c4::Game`, defined in `cpp/include/games/connect4/Game.hpp`. The overall structure looks like this:

```
struct Game {
  struct Constants { ... };
  struct BaseState { ... };

  using FullState = ...;
  using TransformList = ...;

  struct Rules { ... };
  struct IO { ... };
  struct InputTensorizor { ... };
  struct TrainingTargets {
    ...
    using List = ...;
  };
};
```

## Constants

The `Constants` class supplies key constants, such as the number of players in the game, and the number of possible actions in the game.
These are needed so that the various tensors/bitsets used in MCTS can have sizes known at compile-time, which helps with runtime efficiency.

## BaseState / FullState

Each game needs a `BaseState` class and a `FullState` class.

The `BaseState` must be a POD-struct (i.e., copyable via `std::memcpy()`). This requirement ensures that the game-log-reader, used during
network training, can efficiently scan to the $k$'th state in a game-log.

In some games, however, a POD-struct cannot fully capture all aspects of the game state needed for rules calculations. For example, in
chess, in order to support the threefold-repetition-rule, we need a dynamic data structure to store previous states.

To this end, we also have a notion of a `FullState`. A `FullState` should be castable to a `BaseState`, but can also include additional
data structures needed for rules calculations. For games without such requirements, `FullState` can simply be the same class as `BaseState`.

The `GameServer` class, which is responsible for tracking the game state and updating it in response to player actions, only maintains a
`FullState`. The MCTS logic maintains a `FullState` (in order to compute legal moves), but also maintains a history vector of `BaseState`'s.
To perform neural network evaluations, the tail of this vector is cast to a `BaseState[]`, which is then converted to a tensor
via `InputTensorizor`.

## TransformList

Each game should specify `TransformList`, a type-list of the symmetries in the game. You can just have a single Identity element in this type-list for
simplicity, but fully specifying all symmetries will generally improve the rate of learning.

Each class of `TransformList` should have a method to transform a `BaseState` and a method to transform a policy tensor.

## Rules

This class contains static methods that encapsulate the rules of the game.

## IO

This class contains static methods used to print human-readable representations of the game-state and of actions.

## InputTensorizor

This class has a key static `tensorize()` method, which effectively accepts a `BaseState[]` as input, and outputs a policy-tensor.
The size of the array is determined by `1 + Games::Constant::kHistorySize`.

This design decision, of having `InputTensorizor` operator on a `BaseState[]`, ensures that the same function can be used for
input tensor construction at runtime and during training. This is because the MCTS logic maintains a vector of `BaseState` objects,
which can be cast to a `BaseState[]`, and because the self-play log files have a `BaseState[]` section on disk (see
[GameLogFormat.md](GameLogFormat.md)).

It should also provide a `eval_key()` method and a `mcts_key()` method, each of which should take a `FullState` as input, and return
a hashable object as output. These are used to support MCTS.

The `eval_key()` method is used as the key for the neural-network-eval cache-map. In a game like chess, simply casting the `FullState`
to a `BaseState` and returning it (or a hash of it) is likely a good choice, even if the neural network accepts recent positions as
part of the input. This is because identical positions deserve to yield identical evaluations, regardless of move transposition.

The `mcts_key()` method is used to look up MCTS nodes in MCGS. In a game like chess, it is good to clear the position-repetition-tracking
data-structure whenever irreversible moves occur, and then to incorporate a hash of that structure in the return value. There are some
useful comments on this by KataGo author [here](https://docs.google.com/document/d/1JbxsoMtr7_qAAkfYynAgpvuarMMJycaL5toXdnqVJoo/edit#heading=h.z7n8wivwulqx).

## TrainingTargets

This class should specify `List`, a type-list of training-targets. Each class of `List` requires a static `tensorize()` method, which accepts
a `GameLogView` as input and returns a tensor as output.

At a minimum, `TrainingTargets::List` should include `PolicyTarget` and `ValueTarget`. Other auxiliary targets can be included as desired.
