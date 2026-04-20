#pragma once

#include "games/kuhn_poker/Constants.hpp"
#include "util/CppUtil.hpp"

#include <cstdint>
#include <functional>

namespace kuhn_poker {

/*
 * InfoSetState is what a single player can observe: their own card plus the public betting history.
 * The opponent's card is hidden.
 */
struct InfoSetState {
  auto operator<=>(const InfoSetState& other) const = default;

  int8_t my_card = -1;  // this player's card (0=J, 1=Q, 2=K)
  int8_t current_player = 0;
  int8_t phase = kDealPhase;
  int8_t num_actions = 0;
  int8_t actions[4] = {};
};

}  // namespace kuhn_poker

namespace std {

template <>
struct hash<kuhn_poker::InfoSetState> : util::PODHash<kuhn_poker::InfoSetState> {};

}  // namespace std
