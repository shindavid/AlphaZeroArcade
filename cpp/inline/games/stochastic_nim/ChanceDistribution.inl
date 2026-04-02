#include "games/stochastic_nim/ChanceDistribution.hpp"

namespace stochastic_nim {

inline ChanceDistribution::ChanceDistribution(const GameState& state) {
  int num_legal_moves = std::min(stochastic_nim::kChanceDistributionSize, state.stones_left + 1);
  tensor_.setZero();

  float cumulative_prob = 0;
  for (int i = 0; i < num_legal_moves; ++i) {
    tensor_(i) = stochastic_nim::kChanceEventProbs[i];
    cumulative_prob += tensor_(i);
  }
  tensor_(num_legal_moves - 1) += 1 - cumulative_prob;
}

inline Move ChanceDistribution::sample(std::mt19937& prng) const {
  return Move(eigen_util::sample(prng, tensor_)[0], stochastic_nim::kChancePhase);
}

inline float ChanceDistribution::get(const Move& move) const { return tensor_(move.index()); }

}  // namespace stochastic_nim
