#pragma once

#include "games/chess/Game.hpp"
#include "util/EigenUtil.hpp"

namespace chess {

struct InputTensorizor {
    static constexpr int kDim0 = kNumPlayers * (1 + Game::Constants::kNumPreviousStatesToEncode);
    using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDim, kBoardDim>>;

  template <util::concepts::RandomAccessIteratorOf<Game::State> Iter>
  static Tensor tensorize(Iter start, Iter cur) {
    throw std::runtime_error("Not implemented");
  }
};

}  // namespace chess
