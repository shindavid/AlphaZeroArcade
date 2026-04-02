#pragma once

#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/Move.hpp"

namespace stochastic_nim {

struct PolicyEncoding {
  using State = Game::State;
  using Shape = Eigen::Sizes<Game::Constants::kNumMoves>;
  using Tensor = eigen_util::FTensor<Shape>;
  static constexpr int kRank = eigen_util::extract_rank_v<Shape>;
  using Index = Eigen::array<Eigen::Index, kRank>;

  static Index to_index(const Move& move) { return Index{move.index()}; }
  static Move to_move(const State& s, const Index& i) { return Move(i[0], s.current_phase); }
};

}  // namespace stochastic_nim
