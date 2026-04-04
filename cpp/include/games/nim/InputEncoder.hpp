#pragma once

#include "core/SimpleInputEncoder.hpp"
#include "games/nim/Game.hpp"
#include "games/nim/InputFrame.hpp"
#include "games/nim/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace nim {

struct InputEncoder : public core::SimpleInputEncoderBase<Game, InputFrame, Symmetries> {
  using Tensor = eigen_util::FTensor<Eigen::Sizes<nim::kStartingStones>>;

  inline Tensor encode(group::element_t sym = group::kIdentity);
};

}  // namespace nim

#include "inline/games/nim/InputEncoder.inl"
