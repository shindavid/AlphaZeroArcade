#include <games/tictactoe/Game.hpp>

namespace tictactoe {

Game::Types::ActionOutcome Game::Rules::apply(FullState& state, core::action_t action) {
  core::seat_index_t current_player = get_current_player(state);

  mask_t piece_mask = mask_t(1) << action;
  state.cur_player_mask ^= state.full_mask;
  state.full_mask |= piece_mask;

  bool win = false;

  mask_t updated_mask = state.full_mask ^ state.cur_player_mask;
  for (mask_t mask : kThreeInARowMasks) {
    if ((mask & updated_mask) == mask) {
      win = true;
      break;
    }
  }

  Types::ValueArray outcome;
  outcome.setZero();
  if (win) {
    outcome(current_player) = 1.0;
    return Types::ActionOutcome(outcome);
  } else if (std::popcount(state.full_mask) == kNumCells) {
    outcome(0) = 0.5;
    outcome(1) = 0.5;
    return Types::ActionOutcome(outcome);
  }

  return Types::ActionOutcome();
}

Game::Types::ActionMask Game::Rules::get_legal_moves(const FullState& state) {
  Types::ActionMask mask;
  mask.set();
  uint64_t u = state.full_mask;
  while (u) {
    int index = std::countr_zero(u);
    mask[index] = false;
    u &= u - 1;
  }
  return mask;
}

void Game::IO::print_state(const BaseState& state, core::action_t last_action,
                           const Types::player_name_array_t* player_names) {
  auto cp = Rules::get_current_player(state);
  mask_t opp_player_mask = state.opponent_mask();
  mask_t o_mask = (cp == kO) ? state.cur_player_mask : opp_player_mask;
  mask_t x_mask = (cp == kX) ? state.cur_player_mask : opp_player_mask;

  char text[] =
      "0 1 2  | | | |\n"
      "3 4 5  | | | |\n"
      "6 7 8  | | | |\n";

  int offset_table[] = {8, 10, 12, 23, 25, 27, 38, 40, 42};
  for (int i = 0; i < kNumCells; ++i) {
    int offset = offset_table[i];
    if (o_mask & (mask_t(1) << i)) {
      text[offset] = 'O';
    } else if (x_mask & (mask_t(1) << i)) {
      text[offset] = 'X';
    }
  }

  printf("%s\n", text);

  if (player_names) {
    printf("X: %s\n", (*player_names)[kX].c_str());
    printf("O: %s\n\n", (*player_names)[kO].c_str());
  }
  std::cout.flush();
}

void Game::IO::print_mcts_results(const Types::PolicyTensor& action_policy,
                                  const Types::SearchResults& results) {
  const auto& valid_actions = results.valid_actions;
  const auto& mcts_counts = results.counts;
  const auto& net_policy = results.policy_prior;
  const auto& win_rates = results.win_rates;
  const auto& net_value = results.value_prior;

  printf("X: %6.3f%% -> %6.3f%%\n", 100 * net_value(tictactoe::kX), 100 * win_rates(tictactoe::kO));
  printf("O: %6.3f%% -> %6.3f%%\n", 100 * net_value(tictactoe::kO), 100 * win_rates(tictactoe::kX));
  printf("\n");

  printf("%4s %8s %8s %8s\n", "Move", "Net", "Count", "MCTS");
  for (int i = 0; i < tictactoe::kNumCells; ++i) {
    if (valid_actions[i]) {
      float count = mcts_counts(i);
      auto action_p = action_policy(i);
      auto net_p = net_policy(i);
      printf("   %d %8.3f %8.3f %8.3f\n", i, net_p, count, action_p);
    } else {
      printf("\n");
    }
  }
}

}  // namespace tictactoe
