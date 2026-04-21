#pragma once

#include "core/BasicTypes.hpp"

namespace kuhn_poker {

constexpr int kNumPlayers = 2;
constexpr int kNumCards = 3;  // Jack=0, Queen=1, King=2
constexpr int kNumDeals = 6;  // P(3,2) = 6 permutations

// Betting actions
constexpr int kCheck = 0;
constexpr int kBet = 1;
constexpr int kFold = 2;
constexpr int kCall = 3;
constexpr int kNumBettingActions = 4;

constexpr int kDealPhase = 0;
constexpr int kBettingPhase = 1;

// Deal table: maps deal index (0-5) to (card_p0, card_p1)
constexpr int kDealTable[kNumDeals][2] = {
  {0, 1},  // J, Q
  {0, 2},  // J, K
  {1, 0},  // Q, J
  {1, 2},  // Q, K
  {2, 0},  // K, J
  {2, 1},  // K, Q
};

constexpr const char* kCardNames[] = {"J", "Q", "K"};
constexpr const char* kActionNames[] = {"check", "bet", "fold", "call"};

}  // namespace kuhn_poker
