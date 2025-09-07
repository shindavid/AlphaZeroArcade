#pragma once

#include "games/nim/Game.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"

namespace nim {

struct InputTensorizor {
    using Tensor = eigen_util::FTensor<Eigen::Sizes<nim::kStartingStones>>;

  template <util::concepts::RandomAccessIteratorOf<Game::State> Iter>
  static Tensor tensorize(Iter start, Iter cur) {
    Tensor tensor;
    tensor.setZero();
    Iter state = cur;

    for (int i = 0; i < state->stones_left; ++i) {
      tensor(nim::kStartingStones - 1 - i) = 1;
    }
    return tensor;
  }
};

}  // namespace nim
