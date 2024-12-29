#include <games/stochastic_nim/PerfectPlayer.hpp>

namespace stochastic_nim {

inline PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(
    const State& state, const ActionMask& valid_actions) {
  util::release_assert(state.current_mode == kPlayerMode,
                       "PerfectPlayer does not support chance mode");
  return strategy_->get_optimal_action(state.stones_left);
}

PerfectStrategy::PerfectStrategy() {
  for (int i = 0; i < stochastic_nim::kStartingStones + 1; i++) {
    state_values_[i] = -1.0;
    optimal_actions_[i] = -1;
  }

  state_values_[0] = 1.0;
  state_values_[1] = 0.8;
  state_values_[2] = 0.5;
  state_values_[3] = 0.0;
  optimal_actions_[1] = 1;
  optimal_actions_[2] = 2;
  optimal_actions_[3] = 3;
  iterate();
}

inline int PerfectStrategy::get_optimal_action(int stones_left) const {
  util::release_assert(
      (stones_left <= stochastic_nim::kStartingStones) && (stones_left > 0),
      "PerfectStrategy does not support more stones than starting stones or less or equal to 0");
  return optimal_actions_[stones_left] - 1;
}

inline void PerfectStrategy::iterate() {
  Eigen::Array<float, stochastic_nim::kChanceDistributionSize, 1> prob_array{
      stochastic_nim::kChanceEventProbs};
  auto reverse_probs = prob_array.reverse().eval();
  for (int stones_left = 4; stones_left <= stochastic_nim::kStartingStones; stones_left++) {
    optimal_actions_[stones_left] =
        stochastic_nim::kMaxStonesToTake -
        eigen_util::argmax(state_values_.segment(stones_left - stochastic_nim::kMaxStonesToTake,
                                                 stochastic_nim::kMaxStonesToTake));

    state_values_[stones_left] =
        1.0 -
        (reverse_probs *
         eigen_util::slice(
             state_values_,
             Eigen::ArrayXi::LinSpaced(stochastic_nim::kChanceDistributionSize,
                                       stones_left - stochastic_nim::kChanceDistributionSize + 1,
                                       stones_left) -
                 optimal_actions_.segment(stones_left - stochastic_nim::kChanceDistributionSize + 1,
                                          stochastic_nim::kChanceDistributionSize))).sum();
  }
}

}  // namespace stochastic_nim

