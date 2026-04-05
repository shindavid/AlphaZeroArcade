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

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.current_player;
}

inline void Game::Rules::apply(State& state, const Move& move) {
  int action = int(move);
  if (action < 0 || action >= nim::kMaxStonesToTake) {
    throw std::invalid_argument("Invalid action: " + std::to_string(action));
  }

  state.stones_left -= action + 1;
  state.current_player = 1 - state.current_player;
}

inline void Game::IO::print_state(std::ostream& os, const State& state, const Move* last_move,
                                  const Types::player_name_array_t* player_names) {
  os << "[" << state.stones_left << ", " << state.current_player << "]" << std::endl;
}

inline std::string Game::IO::compact_state_repr(const State& state) {
  std::ostringstream ss;
  ss << "[" << state.stones_left << ", " << state.current_player << "]";
  return ss.str();
}

inline Game::Rules::Result Game::Rules::analyze(const State& state) {
  if (state.stones_left == 0) {
    GameOutcome outcome;
    for (int s = 0; s < Constants::kNumPlayers; ++s) {
      outcome[s].share = (s != state.current_player) ? 1.0f : 0.0f;
    }
    return outcome;
  }

  MoveSet legal_moves;
  int n = std::min(state.stones_left, nim::kMaxStonesToTake);
  for (int i = 0; i < n; ++i) {
    legal_moves.add(Move(i));
  }
  return legal_moves;
}

}  // namespace nim
