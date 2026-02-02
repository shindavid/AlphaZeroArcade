#pragma once

#include "core/InputTensorizor.hpp"
#include "games/othello/Game.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace othello {

struct InputTensorizor : public core::SimpleInputTensorizorBase<Game>{
  // +1 for stable discs feature
  static constexpr int kDim0 = kNumPlayers * kNumStatesToEncode + 1;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;

  inline Tensor tensorize(group::element_t sym);
};

}  // namespace othello

#include "inline/games/othello/InputTensorizor.inl"
