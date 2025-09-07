#include "games/chess/Game.hpp"

namespace chess {

void Game::IO::print_state(std::ostream&, const State&, core::action_t last_action,
                           const Types::player_name_array_t* player_names) {
  throw std::runtime_error("Not implemented");
}

}  // namespace chess
