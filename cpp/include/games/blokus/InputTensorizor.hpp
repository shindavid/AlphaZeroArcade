#pragma once

#include "core/SimpleInputTensorizor.hpp"
#include "games/blokus/Game.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace blokus {

struct InputTensorizor : public core::SimpleInputTensorizorBase<Game> {
  // +1 to record the partial move if necessary.
  static constexpr int kDim0 = kNumPlayers * kNumStatesToEncode + 1;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;

  inline Tensor tensorize(group::element_t sym = group::kIdentity);
};

}  // namespace blokus

#include "inline/games/blokus/InputTensorizor.inl"
