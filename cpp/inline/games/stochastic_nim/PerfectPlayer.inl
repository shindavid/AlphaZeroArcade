#include <games/stochastic_nim/PerfectPlayer.hpp>

namespace stochastic_nim {

inline PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(
    const State& state, const ActionMask& valid_actions) {
  util::release_assert(state.current_mode == kPlayerMode,
                       "PerfectPlayer does not support chance mode");
  util::release_assert(state.stones_left <= stochastic_nim::kStartingStones,
                       "PerfectPlayer does not support more stones than starting stones");
  return strategy_->get_optimal_action(state.stones_left);
}

PerfectStrategy::PerfectStrategy() {
  for (int i = 0; i < stochastic_nim::kStartingStones + 1; i++) {
    state_values_[i] = -1.0;
    optimal_actions_[i] = -1;
  }

  state_values_[0] = 1.0;
  iterate();
}

inline void PerfectStrategy::iterate() {
  for (int stones_left = 1; stones_left <= stochastic_nim::kStartingStones; stones_left++) {
    int m = std::min(stones_left, stochastic_nim::kMaxStonesToTake);
    optimal_actions_[stones_left] =
        m - argmax(state_values_.segment(
                std::max(0, stones_left - stochastic_nim::kMaxStonesToTake), m));

    float state_value = 0.0;
    for (int chance_remove = 0; chance_remove < stochastic_nim::kChanceDistributionSize; chance_remove++) {
      int next_stones = stones_left - chance_remove;
      if (next_stones <= 0) {
        state_value += stochastic_nim::kChanceEventProbs[chance_remove] * 1.0;
      } else {
        state_value += stochastic_nim::kChanceEventProbs[chance_remove] *
                       (1.0 - state_values_[next_stones - optimal_actions_[next_stones]]);
      }
    }
    state_values_[stones_left] = state_value;
  }
}

inline int PerfectStrategy::argmax(const Eigen::Array<float, Eigen::Dynamic, 1>& segment) {
  Eigen::Index maxIndex;
  segment.maxCoeff(&maxIndex);
  return static_cast<int>(maxIndex);
}

}  // namespace stochastic_nim

