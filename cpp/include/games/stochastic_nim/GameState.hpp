#pragma once

#include "core/BasicTypes.hpp"

#include <functional>

namespace stochastic_nim {

struct GameState {
  auto operator<=>(const GameState& other) const = default;
  size_t hash() const;

  int stones_left = 0;
  int current_player = 0;
  int last_player = -1;
  core::game_phase_t current_phase = 0;
};

}  // namespace stochastic_nim

namespace std {

template <>
struct hash<stochastic_nim::GameState> {
  size_t operator()(const stochastic_nim::GameState& pos) const { return pos.hash(); }
};
}  // namespace std

#include "inline/games/stochastic_nim/GameState.inl"
