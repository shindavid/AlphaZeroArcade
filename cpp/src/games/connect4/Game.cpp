#include <games/connect4/Game.hpp>

#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <util/AnsiCodes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

namespace c4 {

Game::ActionOutcome Game::Rules::apply(FullState& state, core::action_t action) {
  StateSnapshot& snapshot = state.current;

  column_t col = action;
  mask_t piece_mask = (snapshot.full_mask + _bottom_mask(col)) & _column_mask(col);
  core::seat_index_t current_player = snapshot.get_current_player();

  snapshot.cur_player_mask ^= snapshot.full_mask;
  snapshot.full_mask |= piece_mask;

  bool win = false;

  constexpr mask_t horizontal_block = 1UL + (1UL << 8) + (1UL << 16) + (1UL << 24);
  constexpr mask_t nw_se_diagonal_block = 1UL + (1UL << 7) + (1UL << 14) + (1UL << 21);
  constexpr mask_t sw_ne_diagonal_block = 1UL + (1UL << 9) + (1UL << 18) + (1UL << 27);

  mask_t masks[] = {
      (piece_mask << 1) - (piece_mask >> 3),      // vertical
      piece_mask * horizontal_block,              // horizontal 1
      (piece_mask >> 8) * horizontal_block,       // horizontal 2
      (piece_mask >> 16) * horizontal_block,      // horizontal 3
      (piece_mask >> 24) * horizontal_block,      // horizontal 4
      piece_mask * nw_se_diagonal_block,          // nw-se diagonal 1
      (piece_mask >> 7) * nw_se_diagonal_block,   // nw-se diagonal 2
      (piece_mask >> 14) * nw_se_diagonal_block,  // nw-se diagonal 3
      (piece_mask >> 21) * nw_se_diagonal_block,  // nw-se diagonal 4
      piece_mask * sw_ne_diagonal_block,          // sw-ne diagonal 1
      (piece_mask >> 9) * sw_ne_diagonal_block,   // sw-ne diagonal 2
      (piece_mask >> 18) * sw_ne_diagonal_block,  // sw-ne diagonal 3
      (piece_mask >> 27) * sw_ne_diagonal_block   // sw-ne diagonal 4
  };

  mask_t updated_mask = snapshot.full_mask ^ snapshot.cur_player_mask;
  for (mask_t mask : masks) {
    // popcount filters out both int overflow and shift-to-zero
    if (((mask & updated_mask) == mask) && std::popcount(mask) == 4) {
      win = true;
      break;
    }
  }

  ValueArray outcome;
  outcome.setZero();
  bool terminal = false;
  if (win) {
    outcome(current_player) = 1.0;
    return ActionOutcome(outcome);
  } else if (std::popcount(snapshot.full_mask) == kNumCells) {
    outcome(0) = 0.5;
    outcome(1) = 0.5;
    return ActionOutcome(outcome);
  }

  return ActionOutcome();
}

void Game::IO::print_snapshot(const StateSnapshot& snapshot, core::action_t last_action,
                              const player_name_array_t* player_names) const {
  if (!util::tty_mode() && last_action > -1) {
    std::string s(2 * last_action + 1, ' ');
    printf("%sx\n", s.c_str());
  }

  column_t blink_column = last_action;
  row_t blink_row = -1;
  if (blink_column >= 0) {
    blink_row = std::countr_one(snapshot.full_mask >> (blink_column * 8)) - 1;
  }
  for (row_t row = kNumRows - 1; row >= 0; --row) {
    row_dump(row, row == blink_row ? blink_column : -1);
  }
  printf("|1|2|3|4|5|6|7|\n\n");
  if (player_names) {
    printf("%s%s%s: %s\n", ansi::kRed(""), ansi::kCircle("R"), ansi::kReset(""),
           (*player_names)[kRed].c_str());
    printf("%s%s%s: %s\n\n", ansi::kYellow(""), ansi::kCircle("Y"), ansi::kReset(""),
           (*player_names)[kYellow].c_str());
  }
  std::cout.flush();
}

void Game::IO::print_mcts_results(const PolicyTensor& action_policy,
                                  const MctsSearchResults& results) {
  const auto& valid_actions = results.valid_actions;
  const auto& mcts_counts = results.counts;
  const auto& net_policy = results.policy_prior;
  const auto& win_rates = results.win_rates;
  const auto& net_value = results.value_prior;

  printf("%s%s%s: %6.3f%% -> %6.3f%%\n", ansi::kRed(""), ansi::kCircle("R"), ansi::kReset(""),
         100 * net_value(c4::kRed), 100 * win_rates(c4::kRed));
  printf("%s%s%s: %6.3f%% -> %6.3f%%\n", ansi::kYellow(""), ansi::kCircle("Y"), ansi::kReset(""),
         100 * net_value(c4::kYellow), 100 * win_rates(c4::kYellow));
  printf("\n");
  printf("%3s %8s %8s %8s\n", "Col", "Net", "Count", "Action");

  int j = 0;
  for (int i = 0; i < c4::kNumColumns; ++i) {
    if (valid_actions[i]) {
      printf("%3d %8.3f %8.3f %8.3f\n", i + 1, net_policy(i), mcts_counts(i), action_policy(i));
      ++j;
    } else {
      printf("%3d\n", i + 1);
    }
  }
}

void Game::IO::print_row(const StateSnapshot& snapshot, row_t row, column_t blink_column) const {
  core::seat_index_t current_player = snapshot.get_current_player();
  const char* cur_color = current_player == kRed ? ansi::kRed("R") : ansi::kYellow("Y");
  const char* opp_color = current_player == kRed ? ansi::kYellow("Y") : ansi::kRed("R");

  for (int col = 0; col < kNumColumns; ++col) {
    int index = _to_bit_index(row, col);
    bool occupied = (1UL << index) & snapshot.full_mask;
    bool occupied_by_cur_player = (1UL << index) & snapshot.cur_player_mask;

    const char* color = occupied ? (occupied_by_cur_player ? cur_color : opp_color) : "";
    const char* c = occupied ? ansi::kCircle("") : " ";

    printf("|%s%s%s%s", col == blink_column ? ansi::kBlink("") : "", color, c,
           occupied ? ansi::kReset("") : "");
  }

  printf("|\n");
}

}  // namespace c4
