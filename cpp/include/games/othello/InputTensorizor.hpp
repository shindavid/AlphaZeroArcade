#pragma once

#include "games/othello/Game.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"

namespace othello {

struct InputTensorizor {
  static constexpr int kNumStatesToEncode = 1;

  // +1 for stable discs feature
  static constexpr int kDim0 = kNumPlayers * kNumStatesToEncode + 1;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;

  template <util::concepts::RandomAccessIteratorOf<Game::State> Iter>
  static inline Tensor tensorize(Iter start, Iter cur);
};

}  // namespace othello

#include "inline/games/othello/InputTensorizor.inl"
