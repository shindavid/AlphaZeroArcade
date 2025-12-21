#pragma once

#include "core/concepts/GameConstantsConcept.hpp"
#include "util/MetaProgramming.hpp"

namespace core {
template <concepts::GameConstants GameConstants>
struct DerivedConstants {
  using kNumActionsPerMode = GameConstants::kNumActionsPerMode;
  static constexpr int kNumActionModes = kNumActionsPerMode::size();
  static constexpr int kMaxNumActions = mp::MaxOf_v<kNumActionsPerMode>;
  static constexpr int kMaxBranchingFactor = GameConstants::kMaxBranchingFactor;

};

}  // namespace core
