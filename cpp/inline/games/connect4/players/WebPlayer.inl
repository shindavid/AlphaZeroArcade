#include "games/connect4/players/WebPlayer.hpp"

#include "games/connect4/Constants.hpp"

namespace c4 {

inline boost::json::object WebPlayer::make_state_update_msg(core::seat_index_t seat,
                                                            const State& state,
                                                            core::action_t last_action,
                                                            core::action_mode_t last_mode) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object msg = base_t::make_state_update_msg(seat, state, last_action, last_mode);
  msg["last_col"] = last_action;
  return msg;
}

}  // namespace c4
