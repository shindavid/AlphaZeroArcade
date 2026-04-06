#pragma once

#include "games/chess/Game.hpp"
#include "games/chess/InputFrame.hpp"
#include "games/chess/Move.hpp"

namespace a0achess {

struct PolicyEncoding {
  using Game = a0achess::Game;
  using Shape = Eigen::Sizes<Game::Constants::kNumMoves>;
  using Tensor = eigen_util::FTensor<Shape>;
  static constexpr int kRank = eigen_util::extract_rank_v<Shape>;
  using Index = Eigen::array<Eigen::Index, kRank>;

  static Index to_index(const InputFrame&, const Move& move);
  static Move to_move(const InputFrame&, const Index& index);
};

}  // namespace a0achess

#include "inline/games/chess/PolicyEncoding.inl"
