#pragma once

#include "games/stochastic_nim/Constants.hpp"
#include "games/stochastic_nim/GameState.hpp"
#include "games/stochastic_nim/Move.hpp"
#include "util/EigenUtil.hpp"

#include <random>

namespace stochastic_nim {

class ChanceDistribution {
 public:
  using Shape = Eigen::Sizes<kChanceDistributionSize>;
  using Tensor = eigen_util::FTensor<Shape>;

  ChanceDistribution(const GameState&);
  Move sample(std::mt19937& prng) const;
  float get(const Move&) const;

 private:
  Tensor tensor_;
};

}  // namespace stochastic_nim

#include "inline/games/stochastic_nim/ChanceDistribution.inl"
