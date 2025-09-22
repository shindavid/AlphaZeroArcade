#include "games/nim/Game.hpp"

namespace nim {

inline size_t Game::State::hash() const {
  auto tuple = std::make_tuple(stones_left, current_player);
  std::hash<decltype(tuple)> hasher;
  return hasher(tuple);
}

inline void Game::Rules::init_state(State& state) {
  state.stones_left = nim::kStartingStones;
  state.current_player = 0;
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  Types::ActionMask mask;

  for (int i = 0; i < nim::kMaxStonesToTake; ++i) {
    mask[i] = i + 1 <= state.stones_left;
  }

  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.current_player;
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  if (action < 0 || action >= nim::kMaxStonesToTake) {
    throw std::invalid_argument("Invalid action: " + std::to_string(action));
  }

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

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  return std::to_string(action + 1);
}

inline void Game::IO::print_state(std::ostream& os, const State& state, core::action_t last_action,
                                  const Types::player_name_array_t* player_names) {
  os << "[" << state.stones_left << ", " << state.current_player << "]" << std::endl;
}

inline std::string Game::IO::compact_state_repr(const State& state) {
  std::ostringstream ss;
  ss << "[" << state.stones_left << ", " << state.current_player << "]";
  return ss.str();
}

}  // namespace nim
