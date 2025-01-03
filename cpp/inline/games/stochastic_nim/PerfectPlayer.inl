#include <games/stochastic_nim/PerfectPlayer.hpp>

namespace stochastic_nim {

inline PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(
    const State& state, const ActionMask& valid_actions) {
  util::release_assert(state.current_mode == kPlayerMode);
  return strategy_->get_optimal_action(state.stones_left);
}

PerfectStrategy::PerfectStrategy() {
  init_boundary_conditions();
  iterate();
}

void PerfectStrategy::init_boundary_conditions() {
  Qa[0] = 0.0;
  Qb[0] = 1.0;
}

inline int PerfectStrategy::get_optimal_action(int stones_left) const {
  util::release_assert((stones_left <= stochastic_nim::kStartingStones) && (stones_left > 0));
  return P[stones_left] - 1;
}

inline void PerfectStrategy::iterate() {
  constexpr int n = stochastic_nim::kStartingStones;
  constexpr int c = stochastic_nim::kChanceDistributionSize;
  Eigen::Vector<float, c> probs{stochastic_nim::kChanceEventProbs};
  auto arange_c = Eigen::ArrayXi::LinSpaced(c, 0, c - 1);
  for (int k = 1; k <= n; k++) {
    int m = std::min(k, stochastic_nim::kMaxStonesToTake);

    P[k] = m - eigen_util::argmax(Qb.segment(k - m, m));
    Qa[k] = Qb[k - P[k]];
    Qb[k] = 1.0 - probs.dot(eigen_util::slice(Qa, (k - arange_c).cwiseMax(0)));
  }
}

}  // namespace stochastic_nim

