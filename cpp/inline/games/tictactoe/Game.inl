#include <games/tictactoe/Game.hpp>

namespace tictactoe {

inline size_t Game::BaseState::hash() const {
  return (size_t(full_mask) << 16) + cur_player_mask;
}

inline void Game::Rules::init_state(FullState& state) {
  state.full_mask = 0;
  state.cur_player_mask = 0;
}

inline core::seat_index_t Game::Rules::get_current_player(const BaseState& state) {
  return std::popcount(state.full_mask) % 2;
}

inline Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(const BaseState* start,
                                                                      const BaseState* cur) {
  core::seat_index_t cp = Rules::get_current_player(*cur);
  Tensor tensor;
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t p = _get_player_at(*cur, row, col);
      tensor(0, row, col) = (p == cp);
      tensor(1, row, col) = (p == 1 - cp);
    }
  }
  return tensor;
}

inline Game::TrainingTargets::OwnershipTarget::Tensor
Game::TrainingTargets::OwnershipTarget::tensorize(const Types::GameLogView& view) {
  Tensor tensor;
  const BaseState& state = *view.cur_pos;
  core::seat_index_t cp = Rules::get_current_player(state);
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t p = _get_player_at(state, row, col);
      int val = (p == -1) ? 0 : ((p == cp) ? 2 : 1);
      tensor(row, col) = val;
    }
  }
  return tensor;
}

inline core::seat_index_t Game::_get_player_at(const BaseState& state, int row, int col) {
  int cp = Rules::get_current_player(state);
  int index = row * kBoardDimension + col;
  bool occupied_by_cur_player = (mask_t(1) << index) & state.cur_player_mask;
  bool occupied_by_any_player = (mask_t(1) << index) & state.full_mask;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

}  // namespace tictactoe
