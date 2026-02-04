#pragma once

#include "core/SimpleInputTensorizor.hpp"
#include "games/nim/Game.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace nim {

struct InputTensorizor : public core::SimpleInputTensorizorBase<Game> {
  using Tensor = eigen_util::FTensor<Eigen::Sizes<nim::kStartingStones>>;

  inline Tensor tensorize(group::element_t sym = group::kIdentity);
};

}  // namespace nim

#include "inline/games/nim/InputTensorizor.inl"
