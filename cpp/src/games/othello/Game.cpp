#include <games/othello/Game.hpp>

#include <algorithm>
#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <util/AnsiCodes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

namespace othello {

std::string Game::IO::action_to_str(core::action_t action) {
  int a = action;
  if (a == kPass) {
    return "PA";
  }
  char s[3];
  s[0] = 'A' + (a % 8);
  s[1] = '1' + (a / 8);
  s[2] = 0;
  return s;
}

// copied from edax-reversi repo - board_next()
Game::Types::ActionOutcome Game::Rules::apply(FullState& state, core::action_t action) {
  if (action == kPass) {
    std::swap(state.cur_player_mask, state.opponent_mask);
    state.cur_player = 1 - state.cur_player;
    state.pass_count++;
    if (state.pass_count == kNumPlayers) {
      return compute_outcome(state);
    }
  } else {
    mask_t flipped = flip[action](state.cur_player_mask, state.opponent_mask);
    mask_t cur_player_mask = state.opponent_mask ^ flipped;

    state.opponent_mask = state.cur_player_mask ^ (flipped | (1ULL << action));
    state.cur_player_mask = cur_player_mask;
    state.cur_player = 1 - state.cur_player;
    state.pass_count = 0;

    if ((state.opponent_mask | state.cur_player_mask) == kCompleteBoardMask) {
      return compute_outcome(state);
    }
  }

  return Types::ActionOutcome();
}

Game::Types::ActionMask Game::Rules::get_legal_moves(const FullState& state) {
  uint64_t mask = get_moves(state.cur_player_mask, state.opponent_mask);
  Types::ActionMask valid_actions;
  uint64_t u = mask;
  while (u) {
    int index = std::countr_zero(u);
    valid_actions[index] = true;
    u &= u - 1;
  }
  valid_actions[kPass] = mask == 0;
  return valid_actions;
}

void Game::IO::print_state(const BaseState& state, core::action_t last_action,
                           const Types::player_name_array_t* player_names) {
  Types::ActionMask valid_actions = Rules::get_legal_moves(state);
  bool display_last_action = last_action >= 0;
  int blink_row = -1;
  int blink_col = -1;
  if (display_last_action && last_action != kPass) {
    blink_row = last_action / kBoardDimension;
    blink_col = last_action % kBoardDimension;
  }
  if (!util::tty_mode() && display_last_action) {
    std::string s(2 * blink_col + 3, ' ');
    printf("%sx\n", s.c_str());
  }
  printf("   A B C D E F G H\n");
  for (int row = 0; row < kBoardDimension; ++row) {
    print_row(state, valid_actions, row, row == blink_row ? blink_col : -1);
  }
  std::cout << std::endl;
  int opponent_disc_count = std::popcount(state.opponent_mask);
  int cur_player_disc_count = std::popcount(state.cur_player_mask);

  int black_disc_count = state.cur_player == kBlack ? cur_player_disc_count : opponent_disc_count;
  int white_disc_count = state.cur_player == kWhite ? cur_player_disc_count : opponent_disc_count;

  printf("Score: Player\n");
  printf("%5d: %s%s%s", black_disc_count, ansi::kBlue(""), ansi::kCircle("*"), ansi::kReset(""));
  if (player_names) {
    printf(" [%s]", (*player_names)[kBlack].c_str());
  }
  printf("\n");

  printf("%5d: %s%s%s", white_disc_count, ansi::kWhite(""), ansi::kCircle("0"), ansi::kReset(""));
  if (player_names) {
    printf(" [%s]", (*player_names)[kWhite].c_str());
  }
  printf("\n");
  std::cout << std::endl;
}

void Game::IO::print_row(const BaseState& state, const Types::ActionMask& valid_actions, row_t row,
                         column_t blink_column) {
  core::seat_index_t current_player = Rules::get_current_player(state);
  const char* cur_color = current_player == kBlack ? ansi::kBlue("*") : ansi::kWhite("0");
  const char* opp_color = current_player == kBlack ? ansi::kWhite("0") : ansi::kBlue("*");

  char prefix = ' ';
  if (!util::tty_mode() && blink_column >= 0) {
    prefix = 'x';
  }
  printf("%c%d", prefix, (int)(row + 1));
  for (int col = 0; col < kBoardDimension; ++col) {
    int index = row * kBoardDimension + col;
    bool valid = valid_actions[index];
    bool occupied_by_cur_player = (1UL << index) & state.cur_player_mask;
    bool occupied_by_opp_player = (1UL << index) & state.opponent_mask;
    bool occupied = occupied_by_cur_player || occupied_by_opp_player;

    const char* color =
        occupied_by_cur_player ? cur_color : (occupied_by_opp_player ? opp_color : "");
    const char* c = occupied ? ansi::kCircle("") : (valid ? "." : " ");

    printf("|%s%s%s%s", col == blink_column ? ansi::kBlink("") : "", color, c,
           occupied ? ansi::kReset("") : "");
  }

  printf("|\n");
}

Game::Types::ValueArray Game::Rules::compute_outcome(const FullState& state) {
  Types::ValueArray outcome;
  outcome.setZero();

  int opponent_count = std::popcount(state.opponent_mask);
  int cur_player_count = std::popcount(state.cur_player_mask);
  if (cur_player_count > opponent_count) {
    outcome(state.cur_player) = 1;
  } else if (opponent_count > cur_player_count) {
    outcome(1 - state.cur_player) = 1;
  } else {
    outcome.setConstant(0.5);
  }

  return outcome;
}

void Game::IO::print_mcts_results(const Types::PolicyTensor& action_policy,
                                  const Types::SearchResults& results) {
  const auto& valid_actions = results.valid_actions;
  const auto& mcts_counts = results.counts;
  const auto& net_policy = results.policy_prior;
  const auto& win_rates = results.win_rates;
  const auto& net_value = results.value_prior;

  printf("%s%s%s: %6.3f%% -> %6.3f%%\n", ansi::kBlue(""), ansi::kCircle("*"), ansi::kReset(""),
         100 * net_value(othello::kBlack), 100 * win_rates(othello::kBlack));
  printf("%s%s%s: %6.3f%% -> %6.3f%%\n", ansi::kWhite(""), ansi::kCircle("0"), ansi::kReset(""),
         100 * net_value(othello::kWhite), 100 * win_rates(othello::kWhite));
  printf("\n");

  auto tuple0 = std::make_tuple(mcts_counts(0), action_policy(0), net_policy(0), 0);
  using tuple_t = decltype(tuple0);
  using tuple_array_t = std::array<tuple_t, othello::kNumGlobalActions>;
  tuple_array_t tuples;
  int i = 0;
  for (int a = 0; a < othello::kNumGlobalActions; ++a) {
    if (valid_actions[a]) {
      tuples[i] = std::make_tuple(mcts_counts(a), action_policy(a), net_policy(a), a);
      i++;
    }
  }

  std::sort(tuples.begin(), tuples.end());
  std::reverse(tuples.begin(), tuples.end());

  int num_rows = 10;
  int num_actions = i;
  printf("%4s %8s %8s %8s\n", "Move", "Net", "Count", "MCTS");
  for (i = 0; i < std::min(num_rows, num_actions); ++i) {
    const auto& tuple = tuples[i];

    float count = std::get<0>(tuple);
    auto action_p = std::get<1>(tuple);
    auto net_p = std::get<2>(tuple);
    int action = std::get<3>(tuple);

    if (action == othello::kPass) {
      printf("%4s %8.3f %8.3f %8.3f\n", "Pass", net_p, count, action_p);
    } else {
      int row = action / othello::kBoardDimension;
      int col = action % othello::kBoardDimension;
      printf("  %c%d %8.3f %8.3f %8.3f\n", 'A' + col, row + 1, net_p, count, action_p);
    }
  }
  for (i = num_actions; i < num_rows; ++i) {
    printf("\n");
  }
}

}  // namespace othello
