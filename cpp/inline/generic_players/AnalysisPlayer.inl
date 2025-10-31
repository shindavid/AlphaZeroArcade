#include "core/WebManager.hpp"
#include "search/VerboseManager.hpp"
#include "generic_players/AnalysisPlayer.hpp"
#include "util/Rendering.hpp"

namespace generic {

template <core::concepts::Game Game>
AnalysisPlayer<Game>::AnalysisPlayer(core::AbstractPlayer<Game>* wrapped_player)
    : wrapped_player_(wrapped_player) {
  this->set_name("AnalysisPlayer(" + wrapped_player_->get_name() + ")");
}

template <core::concepts::Game Game>
bool generic::AnalysisPlayer<Game>::start_game() {
  action_ = -1;
  resign_ = false;

  auto* manager = core::WebManager<Game>::get_instance();
  manager->wait_for_connection();

  core::HandlerFuncMap handler_map = {
    {"make_move", [this](const boost::json::object& payload) {this->set_action(payload); }},
    {"resign", [this](const boost::json::object& payload) {this->set_resign(payload); }},
  };
  manager->register_handler(this->get_my_seat(), std::move(handler_map));

  bool started = wrapped_player_->start_game();
  if (!manager->become_starter()) {
    manager->wait_for_new_game_ready();
    manager->send_start_game(make_start_game_msg());
  }
  return started;
}

template <core::concepts::Game Game>
boost::json::object AnalysisPlayer<Game>::make_start_game_msg() {
  using IO = Game::IO;
  util::Rendering::Guard guard(util::Rendering::kText);

  State state;
  Game::Rules::init_state(state);

  boost::json::object payload;
  payload["board"] = IO::state_to_json(state);
  auto seat_assignments = boost::json::array();
  auto player_names = boost::json::array();
  for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
    std::string seat = std::string(1, Game::Constants::kSeatChars[p]);
    seat_assignments.push_back(boost::json::value(seat));
    player_names.push_back(boost::json::value(this->get_player_names()[p]));
  }
  payload["my_seat"] = std::string(1, Game::Constants::kSeatChars[this->get_my_seat()]);
  payload["seat_assignments"] = seat_assignments;
  payload["player_names"] = player_names;

  boost::json::object msg;
  msg["type"] = "start_game";
  msg["payload"] = payload;
  return msg;
}

template <core::concepts::Game Game>
typename Game::Types::ActionResponse AnalysisPlayer<Game>::get_action_response(const ActionRequest& request) {
  auto wrapped_player_response = wrapped_player_->get_action_response(request);

  if (resign_) {
    return ActionResponse::resign();
  }
  if (action_ != -1) {
    core::action_t action = action_;
    action_ = -1;
    return action;
  }

  if (wrapped_player_response.yield_instruction == core::kYield) {
    return wrapped_player_response;
  }

  core::action_t proposed_action = wrapped_player_response.action;

  send_action_request(request.valid_actions, proposed_action);
  notification_unit_ = request.notification_unit;

  return ActionResponse::yield();
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::send_action_request(const ActionMask& valid_actions, core::action_t action) {
  boost::json::object msg;
  msg["type"] = "action_request";
  msg["payload"] = make_action_request_msg(valid_actions, action);
  auto* web_manager = core::WebManager<Game>::get_instance();
  web_manager->send_msg(msg);
}

template <core::concepts::Game Game>
boost::json::object AnalysisPlayer<Game>::make_action_request_msg(const ActionMask& valid_actions,
                                                                  core::action_t action) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::array legal_move_indices;
  for (int i : valid_actions.on_indices()) {
    legal_move_indices.push_back(i);
  }

  boost::json::object payload;
  payload["legal_moves"] = legal_move_indices;
  payload["seat"] = std::string(1, Game::Constants::kSeatChars[this->get_my_seat()]);
  payload["proposed_action"] = action;

  const auto* verbose_data = VerboseManager::get_instance()->verbose_data();
  if (verbose_data) {
    payload["verbose_info"] = verbose_data->to_json();
  }

  return payload;
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::set_action(const boost::json::object& payload) {
  action_ = static_cast<core::action_t>(payload.at("index").as_int64());
  notification_unit_.yield_manager->notify(notification_unit_);
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::set_resign(const boost::json::object& payload) {
  resign_ = true;
  notification_unit_.yield_manager->notify(notification_unit_);
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::receive_state_change(core::seat_index_t seat, const State& state,
                                                core::action_t action) {
  wrapped_player_->receive_state_change(seat, state, action);
  send_state_update(seat, state, action, Game::Rules::get_action_mode(state));
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::send_state_update(core::seat_index_t seat, const State& state,
                                        core::action_t last_action, core::action_mode_t last_mode) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object msg;
  msg["type"] = "state_update";
  msg["payload"] = make_state_update_msg(seat, state, last_action, last_mode);

  auto* web_manager = core::WebManager<Game>::get_instance();
  web_manager->send_msg(msg);
}

template <core::concepts::Game Game>
boost::json::object AnalysisPlayer<Game>::make_state_update_msg(core::seat_index_t seat,
                                                           const State& state,
                                                           core::action_t last_action,
                                                           core::action_mode_t last_mode) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object payload;
  payload["board"] = Game::IO::state_to_json(state);
  payload["seat"] = Game::IO::player_to_str(seat);
  payload["last_action"] = Game::IO::action_to_str(last_action, last_mode);

  const VerboseManager* manager = VerboseManager::get_instance();
  const auto* verbose_data = manager->verbose_data();
  if (verbose_data) {
    payload["verbose_info"] = verbose_data->to_json();
  }

  return payload;
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::end_game(const State& state, const GameResultTensor& outcome) {
  wrapped_player_->end_game(state, outcome);
  boost::json::object msg;
  msg["type"] = "game_end";
  msg["payload"] = make_result_msg(state, outcome);

  auto* web_manager = core::WebManager<Game>::get_instance();
  web_manager->send_msg(msg);
  web_manager->clear_starter();
}

template <core::concepts::Game Game>
boost::json::object AnalysisPlayer<Game>::make_result_msg(const State& state,
                                                          const GameResultTensor& outcome) {
  util::Rendering::Guard guard(util::Rendering::kText);

  boost::json::object payload;
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

}  // namespace generic
