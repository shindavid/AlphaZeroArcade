#include "games/kuhn_poker/ChanceDistribution.hpp"

namespace kuhn_poker {

inline ChanceDistribution::ChanceDistribution(const GameState&) {
  // Uniform distribution over all 6 deals
  tensor_.setConstant(1.0f / kNumDeals);
}

inline Move ChanceDistribution::sample(std::mt19937& prng) const {
  return Move(eigen_util::sample(prng, tensor_)[0], kDealPhase);
}

inline float ChanceDistribution::get(const Move& move) const {
  return tensor_(move.index());
}

}  // namespace kuhn_poker
