#pragma once

#include "core/MultiStateInputTensorizor.hpp"
#include "games/chess/Game.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace chess {

struct InputTensorizor : public core::MultiStateInputTensorizorBase<Game, kNumPastStatesToEncode> {
  // +1 for current state
  static constexpr int kNumStatesToEncode = kNumPastStatesToEncode + 1;
  // 6 (Us pieces) + 6 (Them pieces) + 1 (repitition) = 13
  static constexpr int kPlanesPerStep = 13;
  // Castling (4) + En Passant (1) + Side to Move (1) + Rule50 (1) + Zero/Constant (1) = 8
  static constexpr int kAuxiliaryPlanes = 8;
  // kDim0 = 112
  static constexpr int kDim0 = kPlanesPerStep * kNumStatesToEncode + kAuxiliaryPlanes;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDim, kBoardDim>>;

  inline Tensor tensorize(group::element_t sym = group::kIdentity) {
    throw std::runtime_error("Not implemented");
  }
};

}  // namespace chess
