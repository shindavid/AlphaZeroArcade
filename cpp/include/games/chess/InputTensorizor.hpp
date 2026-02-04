#pragma once

#include "core/MultiStateInputTensorizor.hpp"
#include "games/chess/Game.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace chess {

struct InputTensorizor : public core::MultiStateInputTensorizorBase<Game, kNumPastStatesToEncode> {
  static constexpr int kDim0 = kNumPlayers * kNumStatesToEncode;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDim, kBoardDim>>;

  inline Tensor tensorize(group::element_t sym = group::kIdentity) {
    throw std::runtime_error("Not implemented");
  }
};

}  // namespace chess
