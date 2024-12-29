#include <games/stochastic_nim/PerfectPlayer.hpp>

namespace stochastic_nim {

inline PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(const State& state,
                                                                 const ActionMask& valid_actions) {
  if (state.current_mode == kChanceMode) {
    throw std::invalid_argument("PerfectPlayer does not support chance mode");
  }

  if (state.stones_left > strategy_->get_params().starting_stones) {
    throw std::invalid_argument("PerfectPlayer does not support more stones than starting stones");
  }

  return strategy_->get_optimal_action()[state.stones_left] - 1;
}

PerfectStrategy::PerfectStrategy(Params params) : params_(params) {
  state_value_ = new float[params_.starting_stones + 1];
  optimal_action_ = new int[params_.starting_stones + 1];

  for (int i = 0; i < params_.starting_stones + 1; i++) {
    state_value_[i] = -1.0;
    optimal_action_[i] = -1;
  }

  state_value_[0] = 1.0;
  iterate();
}

inline void PerfectStrategy::iterate() {
  for (int stones_left = 1; stones_left <= params_.starting_stones; stones_left++) {
    float best_value = -1.0;
    int best_move = -1;
    for (int move = 1; move <= params_.max_stones_to_take; move++) {
      int next_stones = stones_left - move;
      if (next_stones <= 0) {
        best_value = 1.0;
        best_move = move;
        break;
      }

      float value = state_value_[next_stones];
      if (value > best_value) {
        best_value = value;
        best_move = move;
      }
    }
    optimal_action_[stones_left] = best_move;

    float state_value = 0.0;
    for (int chance_remove = 0; chance_remove < params_.num_chance_events; chance_remove++) {
      int next_stones = stones_left - chance_remove;
      if (next_stones <= 0) {
        state_value += params_.chance_event_probs[chance_remove] * 1.0;
      } else {
        state_value += params_.chance_event_probs[chance_remove] *
                       (1.0 - state_value_[next_stones - optimal_action_[next_stones]]);
      }
    }
    state_value_[stones_left] = state_value;
  }
}

}  // namespace stochastic_nim

