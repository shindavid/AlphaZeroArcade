#include "games/tictactoe/Game.hpp"

namespace tictactoe {

inline size_t Game::State::hash() const { return (size_t(full_mask) << 16) + cur_player_mask; }

inline core::seat_index_t Game::State::get_player_at(int row, int col) const {
  int cp = Rules::get_current_player(*this);
  int index = row * kBoardDimension + col;
  bool occupied_by_cur_player = (mask_t(1) << index) & cur_player_mask;
  bool occupied_by_any_player = (mask_t(1) << index) & full_mask;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

inline void Game::Rules::init_state(State& state) {
  state.full_mask = 0;
  state.cur_player_mask = 0;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return std::popcount(state.full_mask) % 2;
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

inline boost::json::value Game::IO::state_to_json(const State& state) {
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
