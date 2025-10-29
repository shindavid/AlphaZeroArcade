#include "core/WebManager.hpp"
#include "generic_players/AnalysisPlayer.hpp"
#include "util/Rendering.hpp"

namespace generic {

template <core::concepts::Game Game>
bool generic::AnalysisPlayer<Game>::start_game() {
  action_ = -1;
  resign_ = false;

  if (first_game_) {
    first_game_ = false;
    core::WebManager<Game>::get_instance()->wait_for_connection();
  } else {
    core::WebManager<Game>::get_instance()->wait_for_new_game_ready();
  }

  bool started = wrapped_player_->start_game();
  send_start_game();
  return started;
}

template <core::concepts::Game Game>
void AnalysisPlayer<Game>::send_start_game() {
  boost::json::object msg;
  msg["type"] = "start_game";
  msg["payload"] = make_start_game_msg();

  core::WebManager<Game>::get_instance()->send_msg(msg);
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
  return payload;
}

}  // namespace generic
