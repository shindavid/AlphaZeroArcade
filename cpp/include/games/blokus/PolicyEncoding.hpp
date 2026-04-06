#pragma once

#include "games/blokus/Game.hpp"
#include "games/blokus/InputFrame.hpp"
#include "games/blokus/Move.hpp"

namespace blokus {

struct PolicyEncoding {
  using Game = blokus::Game;
  using Shape = Eigen::Sizes<Game::Constants::kNumMoves>;
  using Tensor = eigen_util::FTensor<Shape>;
  static constexpr int kRank = eigen_util::extract_rank_v<Shape>;
  using Index = Eigen::array<Eigen::Index, kRank>;

  static Index to_index(const InputFrame&, const Move& move) { return Index{move.index()}; }
  static Move to_move(const InputFrame& input_frame, const Index& index) {
    return Move(index[0], input_frame.phase());
  }
};

}  // namespace blokus
