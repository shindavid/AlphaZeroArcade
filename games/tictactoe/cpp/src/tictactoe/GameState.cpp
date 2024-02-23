#include <games/tictactoe/GameState.hpp>

#include <core/SquareBoardSymmetries.hpp>

namespace tictactoe {

GameState::GameOutcome GameState::apply_move(const Action& action) {
  int action_index = action[0];
  core::seat_index_t current_player = get_current_player();

  mask_t piece_mask = mask_t(1) << action_index;
  cur_player_mask_ ^= full_mask_;
  full_mask_ |= piece_mask;

  bool win = false;

  mask_t updated_mask = full_mask_ ^ cur_player_mask_;
  for (mask_t mask : kThreeInARowMasks) {
    if ((mask & updated_mask) == mask) {
      win = true;
      break;
    }
  }

  GameOutcome outcome;
  outcome.setZero();
  if (win) {
    outcome(current_player) = 1.0;
  } else if (std::popcount(full_mask_) == kNumCells) {
    outcome(0) = 0.5;
    outcome(1) = 0.5;
  }

  return outcome;
}

GameState::ActionMask GameState::get_valid_actions() const {
  ActionMask mask;
  mask.setConstant(1);
  uint64_t u = full_mask_;
  while (u) {
    int index = std::countr_zero(u);
    mask(index) = false;
    u &= u - 1;
  }
  return mask;
}

void GameState::dump(const Action* last_action,
                            const player_name_array_t* player_names) const {
  auto cp = get_current_player();
  mask_t opp_player_mask = get_opponent_mask();
  mask_t o_mask = (cp == kO) ? cur_player_mask_ : opp_player_mask;
  mask_t x_mask = (cp == kX) ? cur_player_mask_ : opp_player_mask;

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

}  // namespace tictactoe

namespace mcts {

void SearchResultsDumper<tictactoe::GameState>::dump(const LocalPolicyArray& action_policy,
                                                            const SearchResults& results) {
  const auto& valid_actions = results.valid_actions;
  const auto& mcts_counts = results.counts;
  const auto& net_policy = results.policy_prior;
  const auto& win_rates = results.win_rates;
  const auto& net_value = results.value_prior;

  printf("X: %6.3f%% -> %6.3f%%\n", 100 * net_value(tictactoe::kX), 100 * win_rates(tictactoe::kO));
  printf("O: %6.3f%% -> %6.3f%%\n", 100 * net_value(tictactoe::kO), 100 * win_rates(tictactoe::kX));
  printf("\n");

  printf("%4s %8s %8s %8s\n", "Move", "Net", "Count", "MCTS");
  int j = 0;
  for (int i = 0; i < tictactoe::kNumCells; ++i) {
    if (valid_actions(j)) {
      float count = mcts_counts(i);
      auto action_p = action_policy(j);
      auto net_p = net_policy(j);
      printf("   %d %8.3f %8.3f %8.3f\n", i, net_p, count, action_p);
      ++j;
    }
  }
  for (; j < tictactoe::kNumCells; ++j) {
    printf("\n");
  }
}

}  // namespace mcts
