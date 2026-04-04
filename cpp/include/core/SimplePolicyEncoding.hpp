#pragma once

#include "core/concepts/GameConcept.hpp"
#include "util/EigenUtil.hpp"

namespace core {

// SimplePolicyEncoding can be used for games where Game::Move can be constructed from an int and
// which can be cast to that same int via operator int(), and where the range of possible int values
// is [0, Game::Constants::kNumMoves). The policy encoding is then simply a flat 1D tensor of length
// Game::Constants::kNumMoves, where the index corresponding to a given move is simply int(move).
template <concepts::Game G>
struct SimplePolicyEncoding {
  using Game = G;
  using State = Game::State;
  using Move = Game::Move;
  using Shape = Eigen::Sizes<Game::Constants::kNumMoves>;
  using Tensor = eigen_util::FTensor<Shape>;
  static constexpr int kRank = eigen_util::extract_rank_v<Shape>;
  using Index = Eigen::array<Eigen::Index, kRank>;

  static Index to_index(const Move& move) { return Index{int(move)}; }
  static Move to_move(const State&, const Index& index) { return Move(index[0]); }
};

}  // namespace core
