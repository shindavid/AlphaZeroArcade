#pragma once

#include "games/connect4/Game.hpp"
#include "generic_players/WebPlayer.hpp"

namespace c4 {

class WebPlayer : public generic::WebPlayer<Game> {
 protected:
  using base_t = generic::WebPlayer<Game>;

  // We override he base-class to also provide the front-end with the row of the last action.
  // This helps with the animation logic.
  virtual boost::json::object make_state_update_msg(core::seat_index_t seat, const State& state,
                                                    core::action_t last_action,
                                                    core::action_mode_t last_mode) override;
};

}  // namespace c4

#include "inline/games/connect4/players/WebPlayer.inl"
