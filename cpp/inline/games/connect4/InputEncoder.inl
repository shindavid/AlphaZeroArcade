#include "games/connect4/InputEncoder.hpp"

namespace c4 {

inline InputEncoder::Tensor InputEncoder::encode(group::element_t sym) {
  InputFrame frame = this->current_frame();
  Symmetries::apply(frame, sym);
  core::seat_index_t cp = frame.get_current_player();

  Tensor tensor;
  tensor.setZero();
  for (int row = 0; row < kNumRows; ++row) {
    for (int col = 0; col < kNumColumns; ++col) {
      core::seat_index_t p = frame.get_player_at(row, col);
      if (p < 0) continue;
      int x = (kNumPlayers + cp - p) % kNumPlayers;
      tensor(x, row, col) = 1;
    }
  }
  return tensor;
}

}  // namespace c4
