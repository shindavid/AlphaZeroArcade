#pragma once

#include "core/SimpleInputTensorizor.hpp"
#include "games/connect4/Game.hpp"
#include "util/EigenUtil.hpp"

namespace c4 {

struct InputTensorizor : public core::SimpleInputTensorizorBase<Game> {
  static constexpr int kDim0 = kNumPlayers * kNumStatesToEncode;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kNumRows, kNumColumns>>;

  inline Tensor tensorize(group::element_t sym = group::kIdentity);
};

}  // namespace c4

#include "inline/games/connect4/InputTensorizor.inl"
