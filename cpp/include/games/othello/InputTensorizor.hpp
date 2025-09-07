#pragma once

#include "games/othello/Game.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"

namespace othello {

struct InputTensorizor {
  static constexpr int kDim0 = kNumPlayers * (1 + Game::Constants::kNumPreviousStatesToEncode);
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;

  template <util::concepts::RandomAccessIteratorOf<Game::State> Iter>
  static Tensor tensorize(Iter start, Iter cur) {
    core::seat_index_t cp = Game::Rules::get_current_player(*cur);
    Tensor tensor;
    tensor.setZero();
    int i = 0;
    Iter state = cur;
    while (true) {
      for (int row = 0; row < kBoardDimension; ++row) {
        for (int col = 0; col < kBoardDimension; ++col) {
          core::seat_index_t p = state->get_player_at(row, col);
          if (p < 0) continue;
          int x = (kNumPlayers + cp - p) % kNumPlayers;
          tensor(i + x, row, col) = 1;
        }
      }
      if (state == start) break;
      state--;
      i += kNumPlayers;
    }
    return tensor;
  }
};

}  // namespace othello
