#pragma once

#include "core/SimpleInputEncoder.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/InputFrame.hpp"
#include "games/stochastic_nim/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace stochastic_nim {

struct InputEncoder : public core::SimpleInputEncoderBase<Game, InputFrame, Symmetries> {
  using Game = stochastic_nim::Game;

  constexpr static int kNumFeatures = stochastic_nim::kStartingStonesBitWidth + 1;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<1, kNumFeatures, 1>>;

  inline Tensor encode(group::element_t sym = group::kIdentity);
};

}  // namespace stochastic_nim

#include "inline/games/stochastic_nim/InputEncoder.inl"
