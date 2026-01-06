#include "generic_players/WebPlayer.hpp"

#include "core/WebManager.hpp"
#include "search/VerboseManager.hpp"
#include "util/Rendering.hpp"

#include <boost/json/array.hpp>
#include <boost/json/object.hpp>

namespace generic {

template <core::concepts::Game Game>
bool WebPlayer<Game>::start_game() {
  this->initialize_game();
  return true;
}

template <core::concepts::Game Game>
core::ActionResponse WebPlayer<Game>::get_action_response(const ActionRequest& request) {
  return get_web_response(request, core::kNullAction);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::receive_state_change(const StateChangeUpdate& update) {
  send_state_update(update);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::end_game(const State& state, const GameResultTensor& outcome) {
  send_result_msg(state, outcome);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::handle_action(const boost::json::object& payload, core::seat_index_t seat) {
  if (seat != this->get_my_seat()) {
    return;
  }
  action_ = payload.at("index").as_int64();
  notification_unit_.yield_manager->notify(notification_unit_);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::handle_resign(core::seat_index_t seat) {
  if (seat != this->get_my_seat()) {
    return;
  }
  resign_ = true;
  notification_unit_.yield_manager->notify(notification_unit_);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::handle_backtrack(core::game_tree_index_t index, core::seat_index_t seat) {
  if (seat != this->get_my_seat()) {
    return;
  }

  backtrack_index_ = index;
  notification_unit_.yield_manager->notify(notification_unit_);
}

template <core::concepts::Game Game>
void WebPlayer<Game>::initialize_game() {
  action_ = -1;
  resign_ = false;

  auto* manager = core::WebManager<Game>::get_instance();
  manager->wait_for_connection();

  core::WebManagerClient::wait_for_new_game();
  send_start_game();
}

template <core::concepts::Game Game>
core::ActionResponse WebPlayer<Game>::get_web_response(
  const ActionRequest& request, const core::ActionResponse& proposed_response) {
  if (resign_) {
    return core::ActionResponse::resign();
  }

  if (backtrack_index_ >= 0) {
    int index = backtrack_index_;
    backtrack_index_ = -1;
    return core::ActionResponse::backtrack(index);
  }

  if (action_ != -1) {
    core::action_t action = action_;
    action_ = -1;
    return action;
  }

  send_action_request(request, proposed_response.get_action());
  notification_unit_ = request.notification_unit;
  return core::ActionResponse::yield();
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_result_msg(const State& state, const GameResultTensor& outcome) {
  Message msg;
  msg.add_payload(make_result_msg(state, outcome));
  msg.send();
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_start_game() {
  Message msg;
  msg.add_payload(make_start_game_msg());
  msg.send();
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_start_game_msg() {
  using IO = Game::IO;
  util::Rendering::Guard guard(util::Rendering::kText);

  State state;
  Game::Rules::init_state(state);

  boost::json::object payload;
  payload["type"] = "start_game";
  payload["board"] = IO::state_to_json(state);
  auto seat_assignments = boost::json::array();
  auto player_names = boost::json::array();
  for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
    std::string seat = std::string(1, Game::IO::kSeatChars[p]);
    seat_assignments.push_back(boost::json::value(seat));
    player_names.push_back(boost::json::value(this->get_player_names()[p]));
  }
  payload["my_seat"] = std::string(1, Game::IO::kSeatChars[this->get_my_seat()]);
  payload["seat_assignments"] = seat_assignments;
  payload["player_names"] = player_names;
  Game::IO::add_render_info(state, payload);

  return payload;
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_action_request(const ActionRequest& request, core::action_t proposed_action) {
  Message msg;
  msg.add_payload(make_action_request_msg(request, proposed_action));
  msg.send();
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_action_request_msg(const ActionRequest& request, core::action_t proposed_action) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::array legal_move_indices;
  for (int i : request.valid_actions.on_indices()) {
    legal_move_indices.push_back(i);
  }

  boost::json::object payload;
  payload["type"] = "action_request";
  payload["board"] = Game::IO::state_to_json(request.state);
  payload["legal_moves"] = legal_move_indices;
  payload["seat"] = this->get_my_seat();
  payload["proposed_action"] = proposed_action;

  const auto* verbose_data = VerboseManager::get_instance()->verbose_data();
  if (verbose_data) {
    payload["verbose_info"] = verbose_data->to_json();
  }

  return payload;
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_state_update(const StateChangeUpdate& update) {
  util::Rendering::Guard guard(util::Rendering::kText);

  Message msg;
  msg.add_payload(this->make_tree_node_msg(update));
  msg.add_payload(this->make_state_update_msg(update));
  msg.send();
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_tree_node_msg(const StateChangeUpdate& update) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object payload;
  payload["type"] = "tree_node";
  payload["index"] = update.index;
  payload["parent_index"] = update.parent_index;
  payload["seat"] = std::string(1, Game::IO::kSeatChars[update.seat]);

  return payload;
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_state_update_msg(const StateChangeUpdate& update) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object payload;
  payload["type"] = "state_update";
  payload["board"] = Game::IO::state_to_json(update.state);
  payload["index"] = update.index;
  payload["last_action"] = update.action;
  payload["mode"] = update.mode;
  Game::IO::add_render_info(update.state, payload);

  return payload;
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_result_msg(const State& state,
                                                     const GameResultTensor& outcome) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object payload;
  payload["type"] = "game_end";

  constexpr int P = Game::Constants::kNumPlayers;

  auto array = Game::GameResults::to_value_array(outcome);

  char result_codes[P + 1];
  for (int p = 0; p < P; ++p) {
    if (array[p] == 1) {
      result_codes[p] = 'W';  // Win
    } else if (array[p] == 0) {
      result_codes[p] = 'L';  // Loss
    } else {
      result_codes[p] = 'D';  // Draw
    }
  }
  result_codes[P] = '\0';  // Null-terminate the string
  payload["result_codes"] = std::string(result_codes);

  return payload;
}

template <core::concepts::Game Game>
void WebPlayer<Game>::Message::send() {
  auto* web_manager = core::WebManager<Game>::get_instance();
  web_manager->send_msg(msg_);
}

}  // namespace generic
