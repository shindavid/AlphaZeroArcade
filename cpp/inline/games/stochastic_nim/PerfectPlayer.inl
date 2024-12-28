#include <games/stochastic_nim/PerfectPlayer.hpp>

namespace stochastic_nim {

PerfectPlayer::PerfectPlayer(const Params& params) : params_(params) {
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
    /*
     * Need to update the state-action tensor for all player nodes first because the chance nodes
     * could have a child that is a player node with the same number of stones left, so depend on
     * the player node's value.
     */
    // Q(pi@n, a) = \sum_{a'} P(a') Q(pi*@(n - a), a')
    update_player_state_action_tensor(stones_left, Player0);
    update_player_state_action_tensor(stones_left, Player1);

    // Q(pi*@n, a) = max_{a'} Q(pi@n, a') for player 0
    update_chance_state_action_tensor(stones_left, Player0);

    // Q(pi*@n, a) = min_{a'} Q(pi@n, a') for player 1
    update_chance_state_action_tensor(stones_left, Player1);
  }
}

void PerfectPlayer::update_player_state_action_tensor(int stones_left,
                                                      core::seat_index_t player) {
  State state(stones_left, player, stochastic_nim::kPlayerMode);
  ActionMask valid_actions = Rules::get_legal_moves(state);
  for (auto action : bitset_util::on_indices(valid_actions)) {
    State next_state = Rules::apply(state, action);

    if (next_state.stones_left == ZeroStones) {
      state_action_tensor_(state.stones_left, state.current_player, state.current_mode, action) =
          state_action_tensor_(next_state.stones_left, next_state.current_player,
                               next_state.current_mode, 0);
      continue;
    }

    ChanceDistribution chance_dist = Rules::get_chance_distribution(next_state);
    ActionMask next_valid_actions = Rules::get_legal_moves(next_state);

    float& action_state_value = state_action_tensor_(state.stones_left, state.current_player,
                                                     stochastic_nim::kPlayerMode, action);
    // s' = pi*@(n - a) or [n - a, i, 1]
    // Q(pi@n, a) = \sum_{a'} P(a'|s') Q(pi*@(n - a), a')
    action_state_value = 0.0;
    for (auto next_action : bitset_util::on_indices(next_valid_actions)) {
      float prob = chance_dist(next_action);
      float next_action_value = state_action_tensor_(
          next_state.stones_left, next_state.current_player, next_state.current_mode, next_action);
      action_state_value += prob * next_action_value;
    }
  }
}

void PerfectPlayer::update_chance_state_action_tensor(int stones_left, core::seat_index_t player) {
  State state(stones_left, player, stochastic_nim::kChanceMode);
  ActionMask valid_actions = Rules::get_legal_moves(state);
  for (auto action : bitset_util::on_indices(valid_actions)) {
    State next_state = Rules::apply(state, action);
    if (next_state.stones_left == ZeroStones) {
      state_action_tensor_(state.stones_left, state.current_player, state.current_mode, action) =
          state_action_tensor_(next_state.stones_left, next_state.current_player,
                               next_state.current_mode, 0);
      continue;
    }

    action_value_t action_value = compute_best_action_value(next_state);
    state_action_tensor_(state.stones_left, state.current_player, state.current_mode, action) =
        action_value.value;
  }
}

/*
 * For player 0:
 * Q(pi*@n, a) = max_{a'} Q(pi@n, a')
 * best_action = argmax_{a'} Q(pi@n, a')
 *
 * For player 1:
 * Q(pi*@n, a) = min_{a'} Q(pi@n, a')
 * best_action = argmin_{a'} Q(pi@n, a')
 */
PerfectPlayer::action_value_t PerfectPlayer::compute_best_action_value(const State& state) {
  if (state.current_mode == stochastic_nim::kChanceMode) {
    throw std::invalid_argument("Cannot compute best action value for chance mode");
  }

  std::function<bool(float, float)> comp;
  float best_value;
  if (state.current_player == Player0) {
    comp = [](float value, float best_value_) -> bool { return value > best_value_; };
    best_value = std::numeric_limits<float>::lowest();
  } else {
    comp = [](float value, float best_value_) -> bool { return value < best_value_; };
    best_value = std::numeric_limits<float>::max();
  }

  ActionMask valid_actions = Rules::get_legal_moves(state);
  core::action_t best_action = 0;
  for (auto action : bitset_util::on_indices(valid_actions)) {
    float state_action_value = state_action_tensor_(
        state.stones_left, state.current_player, state.current_mode, action);
    if (comp(state_action_value, best_value)) {
      best_value = state_action_value;
      best_action = action;
    }
  }
  return {best_action, best_value};
}

PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(const State& state,
                                                                 const ActionMask& valid_actions) {
  // TODO: add random move
  action_value_t action_value = compute_best_action_value(state);
  return {action_value.action};
}

}  // namespace stochastic_nim

