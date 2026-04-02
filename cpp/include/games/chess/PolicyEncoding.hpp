#pragma once

#include "games/chess/Game.hpp"
#include "games/chess/Move.hpp"
#include "util/Exceptions.hpp"

namespace a0achess {

struct PolicyEncoding {
  using State = Game::State;
  using Shape = Eigen::Sizes<Game::Constants::kNumMoves>;
  using Tensor = eigen_util::FTensor<Shape>;
  static constexpr int kRank = eigen_util::extract_rank_v<Shape>;
  using Index = Eigen::array<Eigen::Index, kRank>;

  static Index to_index(const Move& move) { throw util::Exception("TODO"); }
  static Move to_move(const State& s, const Index& i) { throw util::Exception("TODO"); }
};

}  // namespace a0achess
