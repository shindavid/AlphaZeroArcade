#include <games/chess/Game.hpp>

namespace chess {

void Game::IO::print_state(std::ostream&, const State&, core::action_t last_action,
                           const Types::player_name_array_t* player_names) {
  throw std::runtime_error("Not implemented");
}
void Game::IO::print_mcts_results(std::ostream&, const Types::PolicyTensorVariant& action_policy,
                                  const Types::SearchResults&) {
  throw std::runtime_error("Not implemented");
}

// TODO: hash sequence of states back up to T-50 or last zeroing move, whichever is closer
Game::InputTensorizor::MCTSKey Game::InputTensorizor::mcts_key(
    const StateHistory& history) {
  throw std::runtime_error("Not implemented");
}

}  // namespace chess
