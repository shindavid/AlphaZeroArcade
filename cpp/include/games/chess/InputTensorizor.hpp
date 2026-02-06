#pragma once

#include "core/MultiStateInputTensorizor.hpp"
#include "games/chess/Game.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

namespace chess {

/*
 * InputTensorizor implements the "Classical" input feature set used by Leela Chess Zero (Lc0).
 * * Dimensions: [112, 8, 8]
 * - History: 8 time steps (Current + 7 past)
 * - Planes per step: 13 (6 Us, 6 Them, 1 Repetitions)
 * - Aux planes: 8 (Castling, Ep, Rule50, etc.)
 *
 * Source:
 * https://github.com/LeelaChessZero/lc0/blob/master/src/neural/encoder.h
 * https://github.com/LeelaChessZero/lc0/blob/master/src/neural/encoder.cc
 */

struct InputTensorizor : public core::MultiStateInputTensorizorBase<Game, kNumPastStatesToEncode> {
  static constexpr int kNumStatesToEncode = kNumPastStatesToEncode + 1;
  static constexpr int kPlanesPerStep = 13;
  static constexpr int kAuxiliaryPlanes = 8;
  static constexpr int kDim0 = kPlanesPerStep * kNumStatesToEncode + kAuxiliaryPlanes;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDim, kBoardDim>>;

  inline Tensor tensorize(group::element_t sym = group::kIdentity) {
    throw std::runtime_error("Not implemented");
  }
};

}  // namespace chess
