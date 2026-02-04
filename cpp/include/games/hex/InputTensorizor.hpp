#pragma once

#include "core/SimpleInputTensorizor.hpp"
#include "games/hex/Game.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace hex {

struct InputTensorizor : public core::SimpleInputTensorizorBase<Game> {
  // +1 for swap-legality plane
  static constexpr int kDim0 = 1 + Constants::kNumPlayers * kNumStatesToEncode;

  using Shape = Eigen::Sizes<kDim0, Constants::kBoardDim, Constants::kBoardDim>;
  using Tensor = eigen_util::FTensor<Shape>;

  inline Tensor tensorize(group::element_t sym = group::kIdentity);
};

}  // namespace hex

#include "inline/games/hex/InputTensorizor.inl"
