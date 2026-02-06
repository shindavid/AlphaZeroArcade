#include "games/othello/InputTensorizor.hpp"

namespace othello {

inline InputTensorizor::Tensor InputTensorizor::tensorize(group::element_t sym) {
  State state = this->state();
  Symmetries::apply(state, sym);
  core::seat_index_t cp = Game::Rules::get_current_player(state);

  Tensor tensor;
  tensor.setZero();
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      tensor(2, row, col) =
        (state.aux.stable_discs & (1ULL << (row * kBoardDimension + col))) ? 1 : 0;
      core::seat_index_t p = state.get_player_at(row, col);
      if (p < 0) continue;
      int x = (kNumPlayers + cp - p) % kNumPlayers;
      tensor(x, row, col) = 1;
    }
  }
  return tensor;
}

}  // namespace othello
