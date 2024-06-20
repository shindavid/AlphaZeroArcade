#pragma once

#include <util/CppUtil.hpp>

namespace core {
namespace concepts {

template <class GC>
concept GameConstants = requires {
  // kNumPlayers is the number of players in the game.
  { util::decay_copy(GC::kNumPlayers) } -> std::same_as<int>;

  // kNumActions is the total number of distinct actions in the game. For go, this is 19*19+1, with
  // the +1 being the pass action.
  { util::decay_copy(GC::kNumActions) } -> std::same_as<int>;

  // kMaxBranchingFactor is an upper-bound on the number of valid actions that can be taken in a
  // single state.
  { util::decay_copy(GC::kMaxBranchingFactor) } -> std::same_as<int>;

  // kHistorySize is the number of previous BaseState's that are needed for the neural network to
  // evaluate the current BaseState. If the neural network does not need any previous BaseState's,
  // kHistorySize should be 0.
  { util::decay_copy(GC::kHistorySize) } -> std::same_as<int>;
};

}  // namespace concepts
}  // namespace core
