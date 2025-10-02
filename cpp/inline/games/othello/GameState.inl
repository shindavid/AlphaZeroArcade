#include "games/othello/GameState.hpp"
#include "games/othello/aux_features/StableDiscs.hpp"

namespace othello {

inline size_t GameState::hash() const {
  auto tuple =
    std::make_tuple(core.opponent_mask, core.cur_player_mask, core.cur_player, core.pass_count);
  std::hash<decltype(tuple)> hasher;
  return hasher(tuple);
}

inline int GameState::get_count(core::seat_index_t seat) const {
  return std::popcount(seat == core.cur_player ? core.cur_player_mask : core.opponent_mask);
}

inline void GameState::compute_aux() {
  aux.stable_discs =
    compute_stable_discs(core.cur_player_mask, core.opponent_mask, aux.stable_discs);
}

inline core::seat_index_t GameState::get_player_at(int row, int col) const {
  int cp = core.cur_player;
  int index = row * kBoardDimension + col;
  bool occupied_by_cur_player = (mask_t(1) << index) & core.cur_player_mask;
  bool occupied_by_opponent = (mask_t(1) << index) & core.opponent_mask;
  return occupied_by_opponent ? (1 - cp) : (occupied_by_cur_player ? cp : -1);
}

}  // namespace othello
