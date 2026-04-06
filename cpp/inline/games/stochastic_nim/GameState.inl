#include "games/stochastic_nim/GameState.hpp"

namespace stochastic_nim {

inline size_t GameState::hash() const {
  auto tuple = std::make_tuple(stones_left, current_player, last_player, phase);
  std::hash<decltype(tuple)> hasher;
  return hasher(tuple);
}

}  // namespace stochastic_nim
