#pragma once

#include "core/SimpleInputTensorizor.hpp"
#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/InputFrame.hpp"
#include "games/tictactoe/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace tictactoe {

struct InputTensorizor : public core::SimpleInputTensorizorBase<Game, InputFrame, Symmetries> {
  static constexpr int kDim0 = kNumPlayers * kNumStatesToEncode;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;

  inline Tensor tensorize(group::element_t sym = group::kIdentity);
};

}  // namespace tictactoe

#include "inline/games/tictactoe/InputTensorizor.inl"
