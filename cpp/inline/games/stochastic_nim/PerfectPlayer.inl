#include <games/stochastic_nim/PerfectPlayer.hpp>

namespace stochastic_nim {

PerfectPlayer::PerfectPlayer(const Params& params) {
  state_action_tensor_.setConstant(-1);
  update_boundary_conditions();
  update_state_action_tensor();
}

void PerfectPlayer::update_boundary_conditions() {
  // Q[0, 0, 0, :] = 0.0
  state_action_tensor_.chip<0>(ZeroStones)
      .chip<0>(Player0)
      .chip<0>(stochastic_nim::kPlayerMode)
      .setConstant(Player1Win);
  // Q[0, 1, 0, :] = 1.0
  state_action_tensor_.chip<0>(ZeroStones)
      .chip<0>(Player1)
      .chip<0>(stochastic_nim::kPlayerMode)
      .setConstant(Player0Win);
  // Q[0, 0, 1, :] = 1.0
  state_action_tensor_.chip<0>(ZeroStones)
      .chip<0>(Player0)
      .chip<0>(stochastic_nim::kChanceMode)
      .setConstant(Player0Win);
  // Q[0, 1, 1, :] = 0.0
  state_action_tensor_.chip<0>(ZeroStones)
      .chip<0>(Player1)
      .chip<0>(stochastic_nim::kChanceMode)
      .setConstant(Player1Win);
}

void PerfectPlayer::update_state_action_tensor() {
  for (int stones_left = 1; (unsigned int)stones_left <= stochastic_nim::kStartingStones;
       ++stones_left) {
    // Need to update the state-action tensor for all player nodes first because the chance nodes
    // depend on the player nodes of the same stones_left.
    for (core::seat_index_t player = 0; player < Constants::kNumPlayers; ++player) {
      // Q(pi@n, a) = \sum_{a'} P(a') Q(pi*@(n - a), a')
      update_player_state_action_tensor(stones_left, player);
    }
    for (core::seat_index_t player = 0; player < Constants::kNumPlayers; ++player) {
      // Q(pi*@n, a) = max_{a'} Q(pi@n, a') for player 0
      // Q(pi*@n, a) = min_{a'} Q(pi@n, a') for player 1
      update_chance_state_action_tensor(stones_left, player);
    }
  }
}

void PerfectPlayer::update_player_state_action_tensor(int stones_left,
                                                      core::seat_index_t player) {
  State state(stones_left, player, stochastic_nim::kPlayerMode);
  ActionMask valid_actions = Rules::get_legal_moves(state);
  for (auto action : bitset_util::on_indices(valid_actions)) {
    State next_state = Rules::apply(state, action);
    if (next_state.stones_left == 0) {
      float next_state_action_value = state_action_tensor_(
          next_state.stones_left, next_state.current_player, next_state.current_mode, 0);
      state_action_tensor_(state.stones_left, state.current_player, state.current_mode, action) =
          next_state_action_value;
      continue;
    }
    ChanceDistribution chance_dist = Rules::get_chance_distribution(next_state);
    ActionMask next_valid_actions = Rules::get_legal_moves(next_state);

    float& action_state_value = state_action_tensor_(state.stones_left, state.current_player,
                                                     stochastic_nim::kPlayerMode, action);
    // s' = pi*@(n - a) or [n - a, i, 1]
    // Q(pi@n, a) = \sum_{a'} P(a'|s') Q(pi*@(n - a), a')
    action_state_value = 0.0;
    for (core::action_t next_action = 0; (size_t)next_action < next_valid_actions.size();
         ++next_action) {
      if (next_valid_actions[next_action]) {
        float prob = chance_dist(next_action);
        float next_action_value =
            state_action_tensor_(next_state.stones_left, next_state.current_player,
                                 next_state.current_mode, next_action);
        action_state_value += prob * next_action_value;
      }
    }
  }
}

void PerfectPlayer::update_chance_state_action_tensor(int stones_left, core::seat_index_t player) {
  State state(stones_left, player, stochastic_nim::kChanceMode);
  ActionMask valid_actions = Rules::get_legal_moves(state);
  for (auto action : bitset_util::on_indices(valid_actions)) {
    State next_state = Rules::apply(state, action);
    if (next_state.stones_left == 0) {
      float next_state_action_value = state_action_tensor_(
          next_state.stones_left, next_state.current_player, next_state.current_mode, 0);
      state_action_tensor_(state.stones_left, state.current_player, state.current_mode, action) =
          next_state_action_value;
      continue;
    }

    ActionMask next_valid_actions = Rules::get_legal_moves(next_state);
    float state_action_value = 0.0;
    state_action_value = compute_action_value(next_valid_actions, next_state);
    state_action_tensor_(state.stones_left, state.current_player, state.current_mode, action) =
        state_action_value;
  }
}

float PerfectPlayer::compute_action_value(const ActionMask& next_valid_actions, const State& next_state) {
  float value = (next_state.current_player == Player0) ? std::numeric_limits<float>::lowest()
                                                       : std::numeric_limits<float>::max();
  std::function<float(float, float)> comp;
  if (next_state.current_player == Player0) {
    comp = [](float x, float y) -> float { return std::max(x, y); };
  } else {
    comp = [](float x, float y) -> float { return std::min(x, y); };
  }

  for (auto next_action : bitset_util::on_indices(next_valid_actions)) {
    float next_action_value = state_action_tensor_(
        next_state.stones_left, next_state.current_player, next_state.current_mode, next_action);
    value = comp(value, next_action_value);
  }
  return value;
}

PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(const State& state,
                                                                 const ActionMask& valid_actions) {
    if (state.current_player == 0) {
      float max_value = std::numeric_limits<float>::lowest();
      core::action_t best_action = 0;
      for (auto action : bitset_util::on_indices(valid_actions)) {
        float action_value = state_action_tensor_(state.stones_left, state.current_player,
                                                  state.current_mode, action);
        if (action_value > max_value) {
          max_value = action_value;
          best_action = action;
        }
      }
      return {best_action};
    } else {
      float min_value = std::numeric_limits<float>::max();
      core::action_t best_action = 0;
      for (auto action : bitset_util::on_indices(valid_actions)) {
        float action_value = state_action_tensor_(state.stones_left, state.current_player,
                                                  state.current_mode, action);
        if (action_value < min_value) {
          min_value = action_value;
          best_action = action;
        }
      }
      return {best_action};
    }
  }

}  // namespace stochastic_nim

