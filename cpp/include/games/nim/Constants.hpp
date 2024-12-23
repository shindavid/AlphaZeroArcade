#pragma once

#include <core/BasicTypes.hpp>

namespace nim {

const int kNumPlayers = 2;
const int kMaxStonesToTake = 3;
const int kStartingStones = 21;

// After each player's turn, a random int r is chose with probability kChanceEventProbs[r],
// and r stones are removed from the pile.
constexpr float kChanceEventProbs[] = {0.5, 0.5};
constexpr int kChanceDistributionSize = sizeof(kChanceEventProbs) / sizeof(float);

const core::action_t kTake1 = 0;
const core::action_t kTake2 = 1;
const core::action_t kTake3 = 2;
}