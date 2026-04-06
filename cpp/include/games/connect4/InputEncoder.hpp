#pragma once

#include "core/SimpleInputEncoder.hpp"
#include "games/connect4/Game.hpp"
#include "games/connect4/InputFrame.hpp"
#include "games/connect4/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace c4 {

struct InputEncoder : public core::SimpleInputEncoderBase<Game, InputFrame, Symmetries> {
  using Game = c4::Game;

  static constexpr int kDim0 = kNumPlayers * kNumFramesToEncode;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kNumRows, kNumColumns>>;

  inline Tensor encode(group::element_t sym = group::kIdentity);
};

}  // namespace c4

#include "inline/games/connect4/InputEncoder.inl"
