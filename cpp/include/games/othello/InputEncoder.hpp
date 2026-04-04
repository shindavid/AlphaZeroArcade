#pragma once

#include "core/SimpleInputEncoder.hpp"
#include "games/othello/Game.hpp"
#include "games/othello/InputFrame.hpp"
#include "games/othello/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace othello {

struct InputEncoder : public core::SimpleInputEncoderBase<Game, InputFrame, Symmetries> {
  // +1 for stable discs feature
  static constexpr int kDim0 = kNumPlayers * kNumFramesToEncode + 1;

  // TODO: we should specialize Keys to only use State::Core for the tranpose-key.

  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;

  inline Tensor encode(group::element_t sym = group::kIdentity);
};

}  // namespace othello

#include "inline/games/othello/InputEncoder.inl"
