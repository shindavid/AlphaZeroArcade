#pragma once

#include "core/DefaultKeys.hpp"
#include "core/InputTensorizor.hpp"
#include "games/connect4/Game.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"

namespace c4 {

struct InputTensorizor {
  static constexpr int kDim0 = kNumPlayers * (1 + Game::Constants::kNumPreviousStatesToEncode);
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kNumRows, kNumColumns>>;

  template <util::concepts::RandomAccessIteratorOf<Game::State> Iter>
  static Tensor tensorize(Iter start, Iter cur);
};

}  // namespace c4

namespace core {

template <>
struct InputTensorizor<c4::Game> : public c4::InputTensorizor {
  using Keys = core::DefaultKeys<c4::Game>;
};

}  // namespace core

#include "inline/games/connect4/InputTensorizor.inl"
