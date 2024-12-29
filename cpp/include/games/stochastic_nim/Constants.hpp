#pragma once

#include <core/BasicTypes.hpp>

#include <bit>

namespace stochastic_nim {

constexpr int kNumPlayers = 2;
constexpr int kMaxStonesToTake = 3;
constexpr unsigned int kStartingStones = 21;

// After each player's turn, a random int r is chose with probability kChanceEventProbs[r],
// and r stones are removed from the pile.
constexpr float kChanceEventProbs[] = {0.2, 0.3, 0.5};
constexpr int kChanceDistributionSize = sizeof(kChanceEventProbs) / sizeof(float);

constexpr core::action_t kTake1 = 0;
constexpr core::action_t kTake2 = 1;
constexpr core::action_t kTake3 = 2;

constexpr core::action_mode_t kPlayerMode = 0;
constexpr core::action_mode_t kChanceMode = 1;

constexpr core::seat_index_t kPlayer0 = 0;
constexpr core::seat_index_t kPlayer1 = 1;

constexpr int kStartingStonesBitWidth = std::bit_width(kStartingStones);
}