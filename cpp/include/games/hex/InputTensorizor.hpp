#pragma once

#include "games/hex/Game.hpp"
#include "util/EigenUtil.hpp"

namespace hex {

struct InputTensorizor {
  // TODO: add more feature planes

  static constexpr int kNumStatesToEncode = 1;

  // +1 for swap-legality plane
  static constexpr int kDim0 = 1 + Constants::kNumPlayers * kNumStatesToEncode;

  using Shape = Eigen::Sizes<kDim0, Constants::kBoardDim, Constants::kBoardDim>;
  using Tensor = eigen_util::FTensor<Shape>;

  template <typename Iter>
  static Tensor tensorize(Iter start, Iter cur) {
    constexpr int B = Constants::kBoardDim;
    constexpr int P = Constants::kNumPlayers;
    core::seat_index_t cp = Game::Rules::get_current_player(*cur);
    Tensor tensor;
    tensor.setZero();
    int i = 0;
    Iter it = cur;
    while (true) {
      Game::State& state = *it;
      for (int p = 0; p < P; ++p) {
        for (int row = 0; row < B; ++row) {
          mask_t mask = state.core.rows[p][row];
          for (; mask; mask &= mask - 1) {
            int col = std::countr_zero(mask);
            int x = (P + cp - p) % P;
            tensor(i + x, row, col) = 1;
          }
        }
      }
      if (it == start) break;
      it--;
      i += P;
    }

    if (cp == Constants::kSecondPlayer && !cur->core.post_swap_phase) {
      // add swap legality plane
      tensor.chip(i, 0).setConstant(1);
    }

    return tensor;
  }
};

}  // namespace hex
