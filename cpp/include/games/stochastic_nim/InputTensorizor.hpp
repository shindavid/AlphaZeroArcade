#pragma once

#include "core/SimpleInputTensorizor.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace stochastic_nim {

struct InputTensorizor : public core::SimpleInputTensorizorBase<Game> {
  constexpr static int kNumFeatures = stochastic_nim::kStartingStonesBitWidth + 1;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<1, kNumFeatures, 1>>;

  inline Tensor tensorize(group::element_t sym = group::kIdentity);
};

}  // namespace stochastic_nim

#include "inline/games/stochastic_nim/InputTensorizor.inl"
