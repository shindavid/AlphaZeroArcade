#include "generic_players/WebPlayer.hpp"

#include <boost/json/array.hpp>

namespace generic {

template <core::concepts::Game Game>
bool WebPlayer<Game>::start_game() {
  this->action_ = -1;
  this->resign_ = false;

  auto* manager = core::WebManager<Game>::get_instance();
  manager->wait_for_connection();

  core::HandlerFuncMap handler_map = {
    {"make_move", [this](const boost::json::object& payload) {this->set_action(payload); }},
    {"resign", [this](const boost::json::object& payload) {this->set_resign(payload); }},
  };
  manager->register_handler(this->get_my_seat(), std::move(handler_map));

  if (manager->become_starter()) {
    manager->wait_for_new_game_ready();
    this->send_start_game();
  }
  return true;
}

template <core::concepts::Game Game>
typename Game::Types::ActionResponse WebPlayer<Game>::get_action_response(const ActionRequest& request) {
  if (this->resign_) {
    return ActionResponse::resign();
  }
  if (this->action_ != -1) {
    core::action_t action = this->action_;
    this->action_ = -1;
    return action;
  }

  this->send_action_request(request.valid_actions, -1);
  this->notification_unit_ = request.notification_unit;

  return ActionResponse::yield();
}

template <core::concepts::Game Game>
void WebPlayer<Game>::receive_state_change(core::seat_index_t seat, const State& state,
                                                core::action_t action) {
  this->send_state_update(seat, state, action, Game::Rules::get_action_mode(state));
}

template <core::concepts::Game Game>
void WebPlayer<Game>::end_game(const State& state, const GameResultTensor& outcome) {
  boost::json::object msg;
  msg["type"] = "game_end";
  msg["payload"] = this->make_result_msg(state, outcome);

  auto* web_manager = core::WebManager<Game>::get_instance();
  web_manager->send_msg(msg);
  web_manager->clear_starter();
  web_manager->clear_handlers();
}

}  // namespace generic
