#pragma once

#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/InputFrame.hpp"
#include "games/stochastic_nim/Move.hpp"

namespace stochastic_nim {

struct PolicyEncoding {
  using Game = stochastic_nim::Game;
  using State = Game::State;
  using Shape = Eigen::Sizes<Game::Constants::kNumMoves>;
  using Tensor = eigen_util::FTensor<Shape>;
  static constexpr int kRank = eigen_util::extract_rank_v<Shape>;
  using Index = Eigen::array<Eigen::Index, kRank>;

  static Index to_index(const InputFrame&, const Move& move) { return Index{move.index()}; }
  static Move to_move(const InputFrame& input_frame, const Index& index) {
    return Move(index[0], input_frame.current_phase);
  }
};

}  // namespace stochastic_nim
