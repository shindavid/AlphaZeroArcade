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

  send_action_request(request.valid_actions, proposed_response.get_action());
  notification_unit_ = request.notification_unit;
  return core::ActionResponse::yield();
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_result_msg(const State& state, const GameResultTensor& outcome) {
  Message msg(Message::BridgeAction::UPDATE);
  msg.add_payload(make_result_msg(state, outcome));
  msg.send();
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_start_game() {
  Message msg(Message::BridgeAction::RESET);
  msg.add_payload(make_start_game_msg());
  msg.send();
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_start_game_msg() {
  using IO = Game::IO;
  util::Rendering::Guard guard(util::Rendering::kText);

  State state;
  Game::Rules::init_state(state);

  Payload payload(Payload::Type::START_GAME);

  payload.add_field("board", IO::state_to_json(state));
  auto seat_assignments = boost::json::array();
  auto player_names = boost::json::array();
  for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
    std::string seat = std::string(1, Game::IO::kSeatChars[p]);
    seat_assignments.push_back(boost::json::value(seat));
    player_names.push_back(boost::json::value(this->get_player_names()[p]));
  }
  payload.add_field("my_seat", std::string(1, Game::IO::kSeatChars[this->get_my_seat()]));
  payload.add_field("seat_assignments", seat_assignments);
  payload.add_field("player_names", player_names);

  auto obj = payload.to_json();
  Game::IO::add_render_info(state, obj);

  return obj;
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_action_request(const ActionMask& valid_actions, core::action_t proposed_action) {
  Message msg(Message::BridgeAction::UPDATE);
  msg.add_payload(make_action_request_msg(valid_actions, proposed_action));
  msg.send();
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_action_request_msg(const ActionMask& valid_actions,
                                                             core::action_t proposed_action) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::array legal_move_indices;
  for (int i : valid_actions.on_indices()) {
    legal_move_indices.push_back(i);
  }

  Payload payload(Payload::Type::ACTION_REQUEST);
  payload.add_field("legal_moves", legal_move_indices);
  payload.add_field("seat", this->get_my_seat());
  payload.add_field("proposed_action", proposed_action);

  const auto* verbose_data = VerboseManager::get_instance()->verbose_data();
  if (verbose_data) {
    payload.add_field("verbose_info", verbose_data->to_json());
  }

  return payload.to_json();
}

template <core::concepts::Game Game>
void WebPlayer<Game>::send_state_update(const StateChangeUpdate& update) {
  util::Rendering::Guard guard(util::Rendering::kText);

  Message msg(Message::BridgeAction::UPDATE);
  msg.add_payload(this->make_tree_node_msg(update));
  msg.add_payload(this->make_state_update_msg(update));
  msg.send();
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_tree_node_msg(const StateChangeUpdate& update) {
  util::Rendering::Guard guard(util::Rendering::kText);

  Payload payload(Payload::Type::TREE_NODE, update.index);
  payload.add_field("index", update.index);
  payload.add_field("parent_index", update.parent_index);
  payload.add_field("seat", std::string(1, Game::IO::kSeatChars[update.seat]));
  return payload.to_json();
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_state_update_msg(const StateChangeUpdate& update) {
  util::Rendering::Guard guard(util::Rendering::kText);

  Payload payload(Payload::Type::STATE_UPDATE);
  payload.add_field("board", Game::IO::state_to_json(update.state));
  payload.add_field("index", update.index);
  payload.add_field("last_action", update.action);
  payload.add_field("mode", update.mode);

  auto obj = payload.to_json();
  Game::IO::add_render_info(update.state, obj);
  return obj;
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::make_result_msg(const State& state,
                                                     const GameResultTensor& outcome) {
  util::Rendering::Guard guard(util::Rendering::kText);

  Payload payload(Payload::Type::GAME_END);
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
  payload.add_field("result_codes", std::string(result_codes));
  return payload.to_json();
}

template <core::concepts::Game Game>
WebPlayer<Game>::Message::Message(BridgeAction bridge_action) {
  msg_["bridge_action"] = (bridge_action == RESET) ? "reset" : "update";
  msg_["payloads"] = boost::json::array();
}

template <core::concepts::Game Game>
void WebPlayer<Game>::Message::send() {
  auto* web_manager = core::WebManager<Game>::get_instance();
  web_manager->send_msg(msg_);
}

template <core::concepts::Game Game>
boost::json::object WebPlayer<Game>::Payload::to_json() const {
  boost::json::object obj = obj_;
  std::string t;
  switch (type_) {
    case START_GAME:
      t = "start_game";
      break;
    case ACTION_REQUEST:
      t = "action_request";
      break;
    case STATE_UPDATE:
      t = "state_update";
      break;
    case GAME_END:
      t = "game_end";
      break;
    case TREE_NODE:
      t = "tree_node";
      break;
  }
  obj["type"] = t;
  obj["cache_key"] = std::format("{}:{}", t, cache_key_index_);
  return obj;
}

template <core::concepts::Game Game>
template <typename T>
void WebPlayer<Game>::Payload::add_field(const std::string& key, T&& value) {
  if (key == "type" || key == "cache_key") {
    throw util::Exception("Cannot add field with reserved key: {}", key);
  }
  obj_[key] = std::forward<T>(value);
}

}  // namespace generic
