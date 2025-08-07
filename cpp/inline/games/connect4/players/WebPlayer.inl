#include "games/connect4/players/WebPlayer.hpp"

#include "games/connect4/Constants.hpp"

namespace c4 {

inline boost::json::object WebPlayer::make_state_update_msg(core::seat_index_t seat,
                                                            const State& state,
                                                            core::action_t last_action,
                                                            core::action_mode_t last_mode) {
  boost::json::object msg = base_t::make_state_update_msg(seat, state, last_action, last_mode);

  boost::json::array col_heights;
  for (int col = 0; col < kNumColumns; ++col) {
    col_heights.push_back(kNumRows - state.num_empty_cells(col));
  }
  msg["col_heights"] = col_heights;
  msg["last_col"] = last_action;
  return msg;
}

}  // namespace c4
