#pragma once

#include "core/SimpleInputEncoder.hpp"
#include "games/blokus/Game.hpp"
#include "games/blokus/InputFrame.hpp"
#include "games/blokus/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace blokus {

struct InputEncoder : public core::SimpleInputEncoderBase<Game, InputFrame, Symmetries> {
  // +1 to record the partial move if necessary.
  static constexpr int kDim0 = kNumPlayers * kNumFramesToEncode + 1;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;

  inline Tensor encode(group::element_t sym = group::kIdentity);
};

}  // namespace blokus

#include "inline/games/blokus/InputEncoder.inl"
