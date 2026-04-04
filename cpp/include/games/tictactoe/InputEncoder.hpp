#pragma once

#include "core/SimpleInputEncoder.hpp"
#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/InputFrame.hpp"
#include "games/tictactoe/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace tictactoe {

struct InputEncoder : public core::SimpleInputEncoderBase<Game, InputFrame, Symmetries> {
  static constexpr int kDim0 = kNumPlayers * kNumFramesToEncode;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;

  inline Tensor encode(group::element_t sym = group::kIdentity);
};

}  // namespace tictactoe

#include "inline/games/tictactoe/InputEncoder.inl"
