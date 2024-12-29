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
  state_values_[stochastic_nim::kStartingStones] = 0.0;
  optimal_actions_[0] = -stochastic_nim::kStartingStones;
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
  for (int stones_left = 1; stones_left <= stochastic_nim::kStartingStones; stones_left++) {
    int num_stones_can_take = std::min(stones_left, stochastic_nim::kMaxStonesToTake);
    optimal_actions_[stones_left] =
        num_stones_can_take - eigen_util::argmax(state_values_.segment(
            std::max(0, stones_left - stochastic_nim::kMaxStonesToTake), num_stones_can_take));

    auto non_neg_stones_left =
        Eigen::ArrayXi::LinSpaced(stochastic_nim::kChanceDistributionSize,
                                  stones_left - stochastic_nim::kChanceDistributionSize + 1,
                                  stones_left).cwiseMax(0);
    state_values_[stones_left] = 1.0 -
        (reverse_probs * eigen_util::slice(state_values_,
        non_neg_stones_left - eigen_util::slice(optimal_actions_, non_neg_stones_left))).sum();
  }
}

}  // namespace stochastic_nim

