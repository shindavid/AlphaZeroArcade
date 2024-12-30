#include <games/stochastic_nim/PerfectPlayer.hpp>

namespace stochastic_nim {

inline PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(
    const State& state, const ActionMask& valid_actions) {
  util::release_assert(state.current_mode == kPlayerMode);
  return strategy_->get_optimal_action(state.stones_left);
}

PerfectStrategy::PerfectStrategy() {
  Qa[0] = -1.0;  // should never be used
  P[0] = -1;  // should never be used
  P[1] = 1;
  P[2] = 2;
  Qa[1] = 1.0;
  Qa[2] = 1.0;
  Qb[0] = 1.0;
  Qb[1] = 0.8;
  Qb[2] = 0.5;
  iterate();
}

inline int PerfectStrategy::get_optimal_action(int stones_left) const {
  util::release_assert((stones_left <= stochastic_nim::kStartingStones) && (stones_left > 0));
  return P[stones_left] - 1;
}

inline void PerfectStrategy::iterate() {
  constexpr int c = stochastic_nim::kChanceDistributionSize;
  constexpr int n = stochastic_nim::kStartingStones;
  constexpr int m = stochastic_nim::kMaxStonesToTake;
  Eigen::Vector<float, c> probs{stochastic_nim::kChanceEventProbs};
  auto rp = probs.reverse();
  for (int k = c; k <= n; k++) {
    P[k] = m - eigen_util::argmax(Qb.segment(k - m, m));
    Qa[k] = Qb[k - P[k]];
    Qb[k] = 1.0 - rp.dot(Qa.segment(k - c + 1, c));
  }
}

}  // namespace stochastic_nim

