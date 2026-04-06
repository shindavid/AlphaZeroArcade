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

  // kMaxBranchingFactor is an upper-bound on the number of valid actions that can be taken in a
  // single state. This is only used for memory allocation purposes. Setting it too high results in
  // merely a mild performance hit, while setting it too low results in a crash.
  { util::decay_copy(GC::kMaxBranchingFactor) } -> std::same_as<int>;
};

}  // namespace concepts
}  // namespace core
