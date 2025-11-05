#pragma once

#include "games/connect4/Game.hpp"
#include "generic_players/WebPlayer.hpp"

namespace c4 {

// We override he base-class to also provide the front-end with col-height info on start_game
// and state_update messages. This information helps with animation logic in the frontend.
class WebPlayer : public generic::WebPlayer<Game> {
 protected:
  using base_t = generic::WebPlayer<Game>;

  virtual boost::json::object make_start_game_msg() override;

  virtual boost::json::object make_state_update_msg(core::seat_index_t seat, const State& state,
                                                    core::action_t last_action,
                                                    core::action_mode_t last_mode) override;

 private:
  void add_col_heights(const State& state, boost::json::object& msg) const;
};

}  // namespace c4

#include "inline/games/connect4/players/WebPlayer.inl"
