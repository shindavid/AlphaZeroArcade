#include "core/WebManager.hpp"
#include "generic_players/AnalysisPlayer.hpp"
#include "util/Rendering.hpp"

namespace generic {

template <core::concepts::Game Game>
bool generic::AnalysisPlayer<Game>::start_game() {
  action_ = -1;
  resign_ = false;

  auto* manager = core::WebManager<Game>::get_instance();
  manager->wait_for_connection();

  bool started = wrapped_player_->start_game();
  if (!manager->has_sent_initial_board()) {
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
  auto* manager = core::WebManager<Game>::get_instance();
  manager->send_msg(msg);
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
  return payload;
}

}  // namespace generic
