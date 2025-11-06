#include "generic_players/AnalysisPlayer.hpp"

#include "core/WebManager.hpp"
#include "util/Rendering.hpp"

namespace generic {

template <core::concepts::Game Game>
bool AnalysisPlayer<Game>::start_game() {
  this->action_ = -1;
  this->resign_ = false;

  auto* manager = core::WebManager<Game>::get_instance();
  manager->wait_for_connection();
  manager->register_client(this->get_my_seat(), this);

  bool started = wrapped_player_->start_game();
  manager->wait_for_new_game_ready();
  this->send_start_game();
  return started;
}

template <core::concepts::Game Game>
typename Game::Types::ActionResponse AnalysisPlayer<Game>::get_action_response(
  const ActionRequest& request) {
  auto wrapped_player_response = wrapped_player_->get_action_response(request);

  if (this->resign_) {
    return ActionResponse::resign();
  }
  if (this->action_ != -1) {
    core::action_t action = this->action_;
    this->action_ = -1;
    return action;
  }

  if (wrapped_player_response.yield_instruction == core::kYield) {
    return wrapped_player_response;
  }

  core::action_t proposed_action = wrapped_player_response.action;

  this->send_action_request(request.valid_actions, proposed_action);
  this->notification_unit_ = request.notification_unit;

  return ActionResponse::yield();
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::receive_state_change(core::seat_index_t seat, const State& state,
                                                core::action_t action) {
  wrapped_player_->receive_state_change(seat, state, action);
  this->send_state_update(seat, state, action, Game::Rules::get_action_mode(state));
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::end_game(const State& state, const GameResultTensor& outcome) {
  wrapped_player_->end_game(state, outcome);
  boost::json::object msg;
  msg["type"] = "game_end";
  msg["payload"] = this->make_result_msg(state, outcome);

  auto* web_manager = core::WebManager<Game>::get_instance();
  web_manager->send_msg(msg);
  web_manager->clear_clients();
}

}  // namespace generic
