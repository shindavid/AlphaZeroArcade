#pragma once

#include "core/SimpleInputTensorizor.hpp"
#include "games/tictactoe/Game.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace tictactoe {

struct InputTensorizor : public core::SimpleInputTensorizorBase<Game> {
  static constexpr int kDim0 = kNumPlayers * kNumStatesToEncode;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;

  inline Tensor tensorize(group::element_t sym = group::kIdentity);
};

}  // namespace tictactoe

#include "inline/games/tictactoe/InputTensorizor.inl"
