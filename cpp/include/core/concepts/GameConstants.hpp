#pragma once

#include <util/CppUtil.hpp>

#include <concepts>

namespace core {
namespace concepts {

template <class GC>
concept GameConstants = requires {
  // kNumPlayers is the number of players in the game.
  { util::decay_copy(GC::kNumPlayers) } -> std::same_as<int>;

  // kNumActions is the total number of distinct actions in the game. For go, this is 19*19+1, with
  // the +1 being the pass action.
  //
  // TODO: kNumActions -> kNumActionsPerType
  // { util::decay_copy(GC::kNumActions) } -> std::same_as<int>;

  // kMaxBranchingFactor is an upper-bound on the number of valid actions that can be taken in a
  // single state. This is only used for memory allocation purposes. Setting it too high results in
  // merely a mild performance hit, while setting it too low results in a crash.
  //
  // TODO: kMaxBranchingFactor -> kMaxBranchingFactorPerType
  // { util::decay_copy(GC::kMaxBranchingFactor) } -> std::same_as<int>;

  // kNumPreviousStatesToEncode is the number of previous State's that are needed for the neural
  // network to evaluate the current State. If the neural network does not need any previous
  // State's, kNumPreviousStatesToEncode should be 0.
  //
  // If using the core::ConstantsBase base-class, this will be 0 by default.
  { util::decay_copy(GC::kNumPreviousStatesToEncode) } -> std::same_as<int>;


};

}  // namespace concepts
}  // namespace core
