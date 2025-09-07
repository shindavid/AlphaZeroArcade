#pragma once

#include "util/CppUtil.hpp"

#include <concepts>

namespace core {
namespace concepts {

template <class GC>
concept GameConstants = requires {
  // The name of the game. Should match GameSpec.name in python.
  { util::decay_copy(GC::kGameName) } -> std::same_as<const char*>;

  // kNumPlayers is the number of players in the game.
  { util::decay_copy(GC::kNumPlayers) } -> std::same_as<int>;

  // kNumActionsPerMode is a sequence of K ints, where K is the number of distinct "action types"
  // in the game. For most games, there is only one action type, and so kNumActionsPerMode consists
  // of a single int.
  //
  // As an example, in the game of Go, there are 19*19+1 actions, with the +1 being the pass action,
  // so kNumActionsPerMode would be util::int_sequence<19*19+1>.
  requires util::concepts::IntSequence<typename GC::kNumActionsPerMode>;

  // kMaxBranchingFactor is an upper-bound on the number of valid actions that can be taken in a
  // single state. This is only used for memory allocation purposes. Setting it too high results in
  // merely a mild performance hit, while setting it too low results in a crash.
  { util::decay_copy(GC::kMaxBranchingFactor) } -> std::same_as<int>;

  // kNumPreviousStatesToEncode is the number of previous State's that are needed for the neural
  // network to evaluate the current State. If the neural network does not need any previous
  // State's, kNumPreviousStatesToEncode should be 0.
  //
  // If using the core::ConstantsBase base-class, this will be 0 by default.
  { util::decay_copy(GC::kNumPreviousStatesToEncode) } -> std::same_as<int>;
};

}  // namespace concepts
}  // namespace core
