#include <games/nim/Game.hpp>

namespace nim {
inline void Game::Rules::init_state(State& state) {
  state.stones_left = kStartingStones;
  state.current_player = 0;
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const StateHistory& history) {
  const State& state = history.current();
  Types::ActionMask mask;

  for (int i = 0; i < nim::kMaxStonesToTake; ++i) {
    mask[i] = i + 1 <= state.stones_left;
  }

  return mask;
}

inline void Game::Rules::apply(StateHistory& history, core::action_t action) {
  if (action < 0 || action >= nim::kMaxStonesToTake) {
    throw std::invalid_argument("Invalid action: " + std::to_string(action));
  }

  State& state = history.extend();
  state.stones_left -= action + 1;
  state.current_player = 1 - state.current_player;
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
  if (state.stones_left == 0) {
    outcome.setZero();
    outcome(last_player) = 1;
    return true;
  }
  return false;
}

}  // namespace nim