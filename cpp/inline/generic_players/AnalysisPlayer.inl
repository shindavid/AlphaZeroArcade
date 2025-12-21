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
core::ActionResponse AnalysisPlayer<Game>::get_action_response(
  const ActionRequest& request) {
  auto proposed_response = wrapped_player_->get_action_response(request);

  if (proposed_response.get_yield_instruction() == core::kYield) {
    return proposed_response;
  }
  return this->get_web_response(request, proposed_response);
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::receive_state_change(const StateChangeUpdate& update) {
  wrapped_player_->receive_state_change(update);
  this->send_state_update(update);
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::end_game(const State& state, const GameResultTensor& outcome) {
  wrapped_player_->end_game(state, outcome);
  this->send_result_msg(state, outcome);
}

}  // namespace generic
