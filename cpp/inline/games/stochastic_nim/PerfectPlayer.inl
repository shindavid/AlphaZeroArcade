#include <games/stochastic_nim/PerfectPlayer.hpp>

namespace stochastic_nim {

inline PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(
    const State& state, const ActionMask& valid_actions) {
  util::release_assert(state.current_mode == kPlayerMode);
  return strategy_->get_optimal_action(state.stones_left);
}

PerfectStrategy::PerfectStrategy() {
  for (int i = 0; i < stochastic_nim::kStartingStones + 1; i++) {
    V[i] = -1.0;
    P[i] = -1;
  }

  V[0] = 1.0;
  V[1] = 0.8;
  V[2] = 0.5;
  V[3] = 0.0;
  P[1] = 1;
  P[2] = 2;
  P[3] = 3;
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
  Eigen::Array<float, c, 1> probs{stochastic_nim::kChanceEventProbs};
  auto rp = probs.reverse();
  for (int k = 4; k <= n; k++) {
    P[k] = m - eigen_util::argmax(V.segment(k - m, m));
    V[k] = 1.0 - (rp * eigen_util::slice(V,
                      Eigen::ArrayXi::LinSpaced(c, k - c + 1, k) - P.segment(k - c + 1, c))).sum();
  }
}

}  // namespace stochastic_nim

