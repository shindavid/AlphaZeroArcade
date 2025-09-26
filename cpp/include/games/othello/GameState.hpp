#pragma once

#include "games/othello/Constants.hpp"

namespace othello {

struct GameState {
  auto operator<=>(const GameState& other) const { return core <=> other.core; }
  bool operator==(const GameState& other) const { return core == other.core; }
  size_t hash() const;
  int get_count(core::seat_index_t seat) const;
  core::seat_index_t get_player_at(int row, int col) const;  // -1 for unoccupied

  struct Core {
    auto operator<=>(const Core& other) const = default;
    bool operator==(const Core& other) const = default;
    mask_t opponent_mask;    // spaces occupied by either player
    mask_t cur_player_mask;  // spaces occupied by current player
    core::seat_index_t cur_player;
    int8_t pass_count;};

  struct Aux {};
  Core core;
  Aux aux;
};

} // namespace othello

#include "inline/games/othello/GameState.inl"
