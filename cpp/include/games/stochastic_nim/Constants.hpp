#pragma once

#include <core/BasicTypes.hpp>

#include <bit>

namespace stochastic_nim {

const int kNumPlayers = 2;
const int kMaxStonesToTake = 3;
const unsigned int kStartingStones = 21;

// After each player's turn, a random int r is chose with probability kChanceEventProbs[r],
// and r stones are removed from the pile.
constexpr float kChanceEventProbs[] = {0.2, 0.3, 0.5};
constexpr int kChanceDistributionSize = sizeof(kChanceEventProbs) / sizeof(float);

const core::action_t kTake1 = 0;
const core::action_t kTake2 = 1;
const core::action_t kTake3 = 2;

const core::action_mode_t kPlayerMode = 0;
const core::action_mode_t kChanceMode = 1;

constexpr int kStartingStonesBitWidth = std::bit_width(kStartingStones);
}