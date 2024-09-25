# Game Concept

To add a new game, you need to supply a `Game` class that adheres to the `core::concepts::Game` concept, defined in `cpp/include/core/concepts/Game.hpp`.

For an example, see `c4::Game`, defined in `cpp/include/games/connect4/Game.hpp`. The overall structure looks like this:

```
struct Game {
  struct Constants { ... };
  struct State { ... };

  using StateHistory = ...;
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

## State / StateHistory

Each game needs a `State` class and a `StateHistory` class.

The `State` class must be a POD-struct. This requirement ensures that we can serialize-to/deserialize-from disk in a straightforward
manner.

The `StateHistory` class maintains a history of recent `State`'s. Recent `State`'s are needed for two reasons:

1. To support rules-calculations (e.g., chess has a repetition-rule and 50-move-rule which demands a history)
2. For neural-network-inference (if the network includes recent states as part of its input)

The exact number of past-states needed is game-specific. If neither of the above reasons imposes a need to store any past states,
then one may use the `core::SimpleStateHistory` class as the `StateHistory`.

## TransformList

Each game should specify `TransformList`, a type-list of the symmetries in the game. You can just have a single Identity element in this type-list for
simplicity, but fully specifying all symmetries will generally improve the rate of learning.

Each class of `TransformList` should have a method to transform a `State` and a method to transform a policy tensor.

## Rules

This class contains static methods that encapsulate the rules of the game.

## IO

This class contains static methods used to print human-readable representations of the game-state and of actions.

## InputTensorizor

This class has a key static `tensorize()` method, which effectively accepts a `State[]` as input, and outputs a policy-tensor.

This design decision, of having `InputTensorizor` operator on a `State[]`, ensures that the same function can be used for
input tensor construction at runtime and during training. This is because the MCTS logic maintains a vector of `State` objects,
which can be cast to a `State[]`, and because the self-play log files have a `State[]` section on disk (see
[GameLogFormat.md](GameLogFormat.md)).

It should also provide a `eval_key()` method and a `mcts_key()` method.

The `eval_key()` method is used as the key for the neural-network-eval cache-map. It accepts the same arguments as `tensorize()`
does. For a game like chess, simply returning the current `State` (or a hash of it) is likely a good choice, even if the neural
network accepts recent positions as part of the input. This is because identical positions deserve to yield identical evaluations,
regardless of move transposition.

The `mcts_key()` method is used to look up MCTS nodes in MCGS. In a game like chess, it is good to clear the position-repetition-tracking
data-structure whenever irreversible moves occur, and then to incorporate a hash of that structure in the return value. There are some
useful comments on this by KataGo author [here](https://docs.google.com/document/d/1JbxsoMtr7_qAAkfYynAgpvuarMMJycaL5toXdnqVJoo/edit#heading=h.z7n8wivwulqx).

## TrainingTargets

This class should specify `List`, a type-list of training-targets. Each class of `List` requires a static `tensorize()` method, which accepts
a `GameLogView` as input and returns a tensor as output.

At a minimum, `TrainingTargets::List` should include `PolicyTarget` and `ValueTarget`. Other auxiliary targets can be included as desired.
