#include "games/connect4/InputTensorizor.hpp"

namespace c4 {

inline InputTensorizor::Tensor InputTensorizor::tensorize(group::element_t sym) {
  State state = this->state();
  Symmetries::apply(state, sym);
  core::seat_index_t cp = Game::Rules::get_current_player(state);

  Tensor tensor;
  tensor.setZero();
  for (int row = 0; row < kNumRows; ++row) {
    for (int col = 0; col < kNumColumns; ++col) {
      core::seat_index_t p = state.get_player_at(row, col);
      if (p < 0) continue;
      int x = (kNumPlayers + cp - p) % kNumPlayers;
      tensor(x, row, col) = 1;
    }
  }
  return tensor;
}

}  // namespace c4
