#pragma once

#include "games/stochastic_nim/Game.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"

namespace stochastic_nim {

struct InputTensorizor {
  // tensor is of the format {binary encoding of stones_left, current_mode}
  constexpr static int kNumFeatures = stochastic_nim::kStartingStonesBitWidth + 1;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<1, kNumFeatures, 1>>;

  template <util::concepts::RandomAccessIteratorOf<Game::State> Iter>
  static Tensor tensorize(Iter start, Iter cur) {
    Tensor tensor;
    tensor.setZero();
    Iter state = cur;

    for (int i = 0; i < stochastic_nim::kStartingStonesBitWidth; ++i) {
      tensor(0, i, 0) = (state->stones_left & (1 << i)) ? 1 : 0;
    }
    tensor(0, stochastic_nim::kStartingStonesBitWidth, 0) = state->current_mode;
    return tensor;
  }
};

}  // namespace stochastic_nim
