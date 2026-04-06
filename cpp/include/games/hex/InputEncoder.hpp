#pragma once

#include "core/SimpleInputEncoder.hpp"
#include "games/hex/Game.hpp"
#include "games/hex/InputFrame.hpp"
#include "games/hex/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace hex {

struct InputEncoder : public core::SimpleInputEncoderBase<Game, InputFrame, Symmetries> {
  using Game = hex::Game;
  using EvalKey = GameState::Core;

  // +1 for swap-legality plane
  static constexpr int kDim0 = 1 + Constants::kNumPlayers * kNumFramesToEncode;

  using Shape = Eigen::Sizes<kDim0, Constants::kBoardDim, Constants::kBoardDim>;
  using Tensor = eigen_util::FTensor<Shape>;

  inline Tensor encode(group::element_t sym = group::kIdentity);
  EvalKey eval_key() const { return current_frame().core; }
};

}  // namespace hex

#include "inline/games/hex/InputEncoder.inl"
