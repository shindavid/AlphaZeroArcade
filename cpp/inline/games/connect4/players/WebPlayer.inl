#include "games/connect4/players/WebPlayer.hpp"

#include "games/connect4/Constants.hpp"

namespace c4 {

inline boost::json::object WebPlayer::make_start_game_msg() {
  util::Rendering::Guard guard(util::Rendering::kText);

  State state;
  Game::Rules::init_state(state);

  boost::json::object msg = base_t::make_start_game_msg();
  add_col_heights(state, msg);

  return msg;
}

inline boost::json::object WebPlayer::make_state_update_msg(core::seat_index_t seat,
                                                            const State& state,
                                                            core::action_t last_action,
                                                            core::action_mode_t last_mode) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object msg = base_t::make_state_update_msg(seat, state, last_action, last_mode);

  this->add_col_heights(state, msg);
  msg["last_col"] = last_action;
  return msg;
}

inline void WebPlayer::add_col_heights(const State& state, boost::json::object& msg) const {
  boost::json::array col_heights;
  for (int col = 0; col < kNumColumns; ++col) {
    col_heights.push_back(kNumRows - state.num_empty_cells(col));
  }
  msg["col_heights"] = col_heights;
}

}  // namespace c4
