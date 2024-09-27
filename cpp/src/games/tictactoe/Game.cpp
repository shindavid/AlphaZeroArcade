#include <games/tictactoe/Game.hpp>

namespace tictactoe {

Game::Types::ActionOutcome Game::Rules::apply(StateHistory& history, core::action_t action) {
  State& state = history.extend();
  core::seat_index_t current_player = get_current_player(state);

  mask_t piece_mask = mask_t(1) << action;
  state.cur_player_mask ^= state.full_mask;
  state.full_mask |= piece_mask;

  util::release_assert(get_current_player(state) != current_player);  // simple sanity check

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

Game::Types::ActionMask Game::Rules::get_legal_moves(const StateHistory& history) {
  const State& state = history.current();
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

void Game::IO::print_state(std::ostream& ss, const State& state, core::action_t last_action,
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

  constexpr int buf_size = 4096;
  char buffer[buf_size];
  int cx = 0;

  cx += snprintf(buffer + cx, buf_size - cx, "%s\n", text);

  if (player_names) {
    cx += snprintf(buffer + cx, buf_size - cx, "X: %s\n", (*player_names)[kX].c_str());
    cx += snprintf(buffer + cx, buf_size - cx, "O: %s\n\n", (*player_names)[kO].c_str());
  }

  util::release_assert(cx < buf_size, "Buffer overflow (%d < %d)", cx, buf_size);
  ss << buffer << std::endl;
}

void Game::IO::print_mcts_results(std::ostream& ss, const Types::PolicyTensor& action_policy,
                                  const Types::SearchResults& results) {
  const auto& valid_actions = results.valid_actions;
  const auto& mcts_counts = results.counts;
  const auto& net_policy = results.policy_prior;
  const auto& win_rates = results.win_rates;
  const auto& net_value = results.value_prior;

  constexpr int buf_size = 4096;
  char buffer[buf_size];
  int cx = 0;

  cx += snprintf(buffer + cx, buf_size - cx, "X: %6.3f%% -> %6.3f%%\n",
                 100 * net_value(tictactoe::kX), 100 * win_rates(tictactoe::kX));
  cx += snprintf(buffer + cx, buf_size - cx, "O: %6.3f%% -> %6.3f%%\n",
                 100 * net_value(tictactoe::kO), 100 * win_rates(tictactoe::kO));
  cx += snprintf(buffer + cx, buf_size - cx, "\n");

  cx += snprintf(buffer + cx, buf_size - cx, "%4s %8s %8s %8s\n", "Move", "Net", "Count", "MCTS");
  int j = 0;
  for (int i = 0; i < tictactoe::kNumCells; ++i) {
    if (valid_actions[i]) {
      float count = mcts_counts(i);
      auto action_p = action_policy(i);
      auto net_p = net_policy(i);
      cx += snprintf(buffer + cx, buf_size - cx, "   %d %8.3f %8.3f %8.3f\n", i, net_p, count,
                     action_p);
      ++j;
    }
  }
  for (; j < tictactoe::kNumCells; ++j) {
    cx += snprintf(buffer + cx, buf_size - cx, "\n");
  }

  util::release_assert(cx < buf_size, "Buffer overflow (%d < %d)", cx, buf_size);
  ss << buffer << std::endl;
}

}  // namespace tictactoe
