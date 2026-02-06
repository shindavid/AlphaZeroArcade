#include "games/hex/InputTensorizor.hpp"

namespace hex {

inline InputTensorizor::Tensor InputTensorizor::tensorize(group::element_t sym) {
  State state = this->state();
  Symmetries::apply(state, sym);
  constexpr int B = Constants::kBoardDim;
  constexpr int P = Constants::kNumPlayers;
  core::seat_index_t cp = Game::Rules::get_current_player(state);

  Tensor tensor;
  tensor.setZero();

  for (int p = 0; p < P; ++p) {
    for (int row = 0; row < B; ++row) {
      mask_t mask = state.core.rows[p][row];
      for (; mask; mask &= mask - 1) {
        int col = std::countr_zero(mask);
        int x = (P + cp - p) % P;
        tensor(x, row, col) = 1;
      }
    }
  }

  if (cp == Constants::kSecondPlayer && !state.core.post_swap_phase) {
    // add swap legality plane
    tensor.chip(0, 0).setConstant(1);
  }

  return tensor;
}

}  // namespace hex
