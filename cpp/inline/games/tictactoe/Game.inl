#include "games/tictactoe/Game.hpp"

namespace tictactoe {

inline void Game::Rules::init_state(State& state) {
  state.full_mask = 0;
  state.cur_player_mask = 0;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.get_current_player();
}

inline std::string Game::IO::compact_state_repr(const State& state) {
  char buf[12];
  const char* syms = "_XO";

  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      buf[row * 4 + col] = syms[state.get_player_at(row, col) + 1];
    }
  }
  buf[3] = '\n';
  buf[7] = '\n';
  buf[11] = '\0';

  return std::string(buf);
}

inline boost::json::value Game::IO::info_set_to_json(const InfoSet& state) {
  char buf[kBoardDimension * kBoardDimension + 1];
  int idx = 0;
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t seat = state.get_player_at(row, col);
      buf[idx++] = (seat == -1) ? ' ' : kSeatChars[seat];
    }
  }
  buf[kBoardDimension * kBoardDimension] = '\0';
  return buf;
}

}  // namespace tictactoe
