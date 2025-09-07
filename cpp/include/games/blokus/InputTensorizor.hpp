#pragma once

#include "games/blokus/Game.hpp"
#include "util/EigenUtil.hpp"

namespace blokus {

struct InputTensorizor {
  // TODO: add unplayed-pieces as an auxiliary input.

  // +1 to record the partial move if necessary.
  static constexpr int kDim0 = kNumPlayers * (1 + Game::Constants::kNumPreviousStatesToEncode) + 1;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDimension, kBoardDimension>>;

  template <typename Iter>
  static Tensor tensorize(Iter start, Iter cur) {
    core::seat_index_t cp = Game::Rules::get_current_player(*cur);
    Tensor tensor;
    tensor.setZero();
    int i = 0;
    Iter state = cur;
    while (true) {
      for (color_t c = 0; c < kNumColors; ++c) {
        color_t rc = (kNumColors + c - cp) % kNumColors;
        for (Location loc : state->core.occupied_locations[c].get_set_locations()) {
          tensor(i + rc, loc.row, loc.col) = 1;
        }
      }
      if (state == start) break;
      state--;
      i += kNumColors;
    }

    if (cur->core.partial_move.valid()) {
      Location loc = cur->core.partial_move;
      tensor(kDim0 - 1, loc.row, loc.col) = 1;
    }
    return tensor;
  }
};

}  // namespace blokus
