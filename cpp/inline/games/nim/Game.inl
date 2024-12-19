#include <games/nim/Game.hpp>

namespace nim {
inline void Game::Rules::init_state(State& state) {
  Game::set_stones(state, kStartingStones);
  Game::set_player(state, 0);
  Game::set_player_ready(state, 1);
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const StateHistory& history) {
  const State& state = history.current();
  Types::ActionMask mask;

  for (int i = 0; i < nim::kMaxStonesToTake; ++i) {
    mask[i] = i + 1 <= Game::get_stones(state);
  }

  return mask;
}

inline void Game::Rules::apply(StateHistory& history, core::action_t action) {
  if (action < 0 || action >= nim::kMaxStonesToTake) {
    throw std::invalid_argument("Invalid action: " + std::to_string(action));
  }

  State& state = history.extend();
  Game::set_stones(state, Game::get_stones(state) - (action + 1));
  Game::set_player(state, 1 - Game::get_player(state));
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
  if (Game::get_stones(state) == 0) {
    outcome.setZero();
    outcome(last_player) = 1;
    return true;
  }
  return false;
}

}  // namespace nim