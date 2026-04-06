#include "games/tictactoe/GameState.hpp"

namespace tictactoe {

inline size_t GameState::hash() const { return (size_t(full_mask) << 16) + cur_player_mask; }

inline core::seat_index_t GameState::get_player_at(int row, int col) const {
  int cp = get_current_player();
  int index = row * kBoardDimension + col;
  bool occupied_by_cur_player = (mask_t(1) << index) & cur_player_mask;
  bool occupied_by_any_player = (mask_t(1) << index) & full_mask;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

}  // namespace tictactoe
