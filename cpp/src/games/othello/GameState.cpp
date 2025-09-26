#include "games/othello/Game.hpp"
#include "games/othello/GameState.hpp"

namespace othello {

core::seat_index_t GameState::get_player_at(int row, int col) const {
  int cp = Game::Rules::get_current_player(*this);
  int index = row * kBoardDimension + col;
  bool occupied_by_cur_player = (mask_t(1) << index) & core.cur_player_mask;
  bool occupied_by_opponent = (mask_t(1) << index) & core.opponent_mask;
  return occupied_by_opponent ? (1 - cp) : (occupied_by_cur_player ? cp : -1);
}

}  // namespace othello
