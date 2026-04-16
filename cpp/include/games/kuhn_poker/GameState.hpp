#pragma once

#include "games/kuhn_poker/Constants.hpp"
#include "util/CppUtil.hpp"

#include <cstdint>
#include <functional>

namespace kuhn_poker {

struct GameState {
  auto operator<=>(const GameState& other) const = default;

  int8_t cards[kNumPlayers] = {-1, -1};  // card per player (0=J, 1=Q, 2=K), -1=not dealt
  int8_t current_player = 0;
  int8_t phase = kDealPhase;
  int8_t num_actions = 0;
  int8_t actions[4] = {};  // betting action history (max 4 actions in Kuhn poker)
};

}  // namespace kuhn_poker

namespace std {

template <>
struct hash<kuhn_poker::GameState> : util::PODHash<kuhn_poker::GameState> {};

}  // namespace std
