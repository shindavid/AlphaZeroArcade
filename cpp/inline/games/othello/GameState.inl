#include "games/othello/GameState.hpp"

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

}  // namespace othello
