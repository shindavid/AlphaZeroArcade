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

}  // namespace generic
