#pragma once

#include "games/kuhn_poker/Constants.hpp"
#include "games/kuhn_poker/GameState.hpp"
#include "games/kuhn_poker/Move.hpp"
#include "util/EigenUtil.hpp"

#include <random>

namespace kuhn_poker {

class ChanceDistribution {
 public:
  using Shape = Eigen::Sizes<kNumDeals>;
  using Tensor = eigen_util::FTensor<Shape>;

  ChanceDistribution(const GameState&);
  Move sample(std::mt19937& prng) const;
  float get(const Move&) const;

 private:
  Tensor tensor_;
};

}  // namespace kuhn_poker

#include "inline/games/kuhn_poker/ChanceDistribution.inl"
