#include "games/tictactoe/InputTensorizor.hpp"

namespace tictactoe {

inline InputTensorizor::Tensor InputTensorizor::tensorize(group::element_t sym) {
  InputFrame frame = this->current_frame();
  Symmetries::apply(frame, sym);
  core::seat_index_t cp = Game::Rules::get_current_player(frame);

  Tensor tensor;
  tensor.setZero();
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t p = frame.get_player_at(row, col);
      if (p < 0) continue;
      int x = (kNumPlayers + cp - p) % kNumPlayers;
      tensor(x, row, col) = 1;
    }
  }
  return tensor;
}

}  // namespace tictactoe
