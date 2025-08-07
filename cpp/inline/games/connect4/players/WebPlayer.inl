#include "games/connect4/players/WebPlayer.hpp"

namespace c4 {

inline boost::json::object WebPlayer::make_state_update_msg(core::seat_index_t seat,
                                                            const State& state,
                                                            core::action_t last_action,
                                                            core::action_mode_t last_mode) {
  boost::json::object msg = base_t::make_state_update_msg(seat, state, last_action, last_mode);

  // Add the row of the last action to the message.
  msg["last_row"] = last_action >= 0 ? state.num_empty_cells(last_action) : -1;
  return msg;
}

}  // namespace c4
