#include "generic_players/AnalysisPlayer.hpp"

#include "core/WebManager.hpp"
#include "util/Rendering.hpp"

namespace generic {

template <core::concepts::Game Game>
bool AnalysisPlayer<Game>::start_game() {
  bool started = wrapped_player_->start_game();
  this->initialize_game();
  return started;
}

template <core::concepts::Game Game>
typename Game::Types::ActionResponse AnalysisPlayer<Game>::get_action_response(
  const ActionRequest& request) {
  auto proposed_response = wrapped_player_->get_action_response(request);

  if (proposed_response.yield_instruction == core::kYield) {
    return proposed_response;
  }
  return this->get_web_response(request, proposed_response);
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::receive_state_change(core::seat_index_t seat, const State& state,
                                                core::action_t action,
                                                core::node_ix_t state_node_ix) {
  wrapped_player_->receive_state_change(seat, state, action, state_node_ix);
  this->send_state_update(seat, state, action, Game::Rules::get_action_mode(state));
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::end_game(const State& state, const GameResultTensor& outcome) {
  wrapped_player_->end_game(state, outcome);
  this->send_result_msg(state, outcome);
}

}  // namespace generic
