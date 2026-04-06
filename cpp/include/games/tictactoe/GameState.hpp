#pragma once

#include "core/BasicTypes.hpp"
#include "games/tictactoe/Constants.hpp"

#include <functional>

namespace tictactoe {

struct GameState {
  auto operator<=>(const GameState& other) const = default;
  size_t hash() const;
  mask_t opponent_mask() const { return full_mask ^ cur_player_mask; }
  core::seat_index_t get_player_at(int row, int col) const;
  core::seat_index_t get_current_player() const { return std::popcount(full_mask) % 2; }

  mask_t full_mask;        // spaces occupied by either player
  mask_t cur_player_mask;  // spaces occupied by current player
};

}  // namespace tictactoe

namespace std {

template <>
struct hash<tictactoe::GameState> {
  size_t operator()(const tictactoe::GameState& pos) const { return pos.hash(); }
};

}  // namespace std

#include "inline/games/tictactoe/GameState.inl"
