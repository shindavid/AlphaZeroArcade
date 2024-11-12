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
  { util::decay_copy(GC::kNumActions) } -> std::same_as<int>;

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

  // kOpeningLength is a subjectively chosen value that indicates the number of moves that are
  // considered to be part of the "opening" phase of the game. This is used to select default
  // parameters for move-temperature and root-softmax-temperature. Those parameters will decay
  // using kOpeningLength as the *quarter-life* (or, 0.5 * kOpeningLength as the half-life).
  //
  // Rougly speaking, if you are willing to tolerate slightly suboptimal move-selection in the first
  // N moves of the game, tightening up to optimal play after that, then you should set
  // kOpeningLength to N. The reason to tolerate suboptimality in the opening is so that the agent
  // does not act too deterministically. This helps achieve data diversity in training, and also
  // more meaningful testing.
  //
  // KataGo effectively uses a value of 38 for this in 19x19 go.
  // { util::decay_copy(GC::kOpeningLength) } -> std::same_as<float>;

  // If kStoreStates is enabled, then the Node will store the State in the Node itself. This can
  // also be enabled by enabling the STORE_STATES macro.
  //
  // If using the core::ConstantsBase base-class, this will be false by default.
  // { util::decay_copy(GC::kStoreStates) } -> std::same_as<bool>;
};

}  // namespace concepts
}  // namespace core
