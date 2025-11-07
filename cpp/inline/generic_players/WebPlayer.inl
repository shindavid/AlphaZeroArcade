#include "generic_players/WebPlayer.hpp"

#include <boost/json/array.hpp>

namespace generic {

template <core::concepts::Game Game>
bool WebPlayer<Game>::start_game() {
  this->initialize_game();
  return true;
}

template <core::concepts::Game Game>
typename Game::Types::ActionResponse WebPlayer<Game>::get_action_response(
  const ActionRequest& request) {
  return this->get_web_response(request, -1);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::receive_state_change(core::seat_index_t seat, const State& state,
                                           core::action_t action) {
  this->send_state_update(seat, state, action, Game::Rules::get_action_mode(state));
}

template <core::concepts::Game Game>
void WebPlayer<Game>::end_game(const State& state, const GameResultTensor& outcome) {
  this->send_result_msg(state, outcome);
}

}  // namespace generic
