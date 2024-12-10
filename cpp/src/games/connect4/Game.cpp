#include <games/connect4/Game.hpp>

#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <util/AnsiCodes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

namespace c4 {

bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                              core::action_t last_action, GameResults::Tensor& outcome) {
  constexpr mask_t horizontal_block = 1UL + (1UL << 8) + (1UL << 16) + (1UL << 24);
  constexpr mask_t nw_se_diagonal_block = 1UL + (1UL << 7) + (1UL << 14) + (1UL << 21);
  constexpr mask_t sw_ne_diagonal_block = 1UL + (1UL << 9) + (1UL << 18) + (1UL << 27);

  column_t col = last_action;
  mask_t piece_mask = ((state.full_mask + _bottom_mask(col)) & (_column_mask(col) << 1)) >> 1;

  util::release_assert(last_player != get_current_player(state));
  util::release_assert(last_action >= 0);
  util::release_assert(std::popcount(piece_mask) == 1);

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

  mask_t updated_mask = state.full_mask ^ state.cur_player_mask;
  for (mask_t mask : masks) {
    // popcount filters out both int overflow and shift-to-zero
    if (((mask & updated_mask) == mask) && std::popcount(mask) == 4) {
      outcome = core::WinLossDrawResults::win(last_player);
      return true;
    }
  }

  if (std::popcount(state.full_mask) == kNumCells) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  return false;
}

void Game::IO::print_state(std::ostream& ss, const State& state, core::action_t last_action,
                           const Types::player_name_array_t* player_names) {
  constexpr int buf_size = 4096;
  char buffer[buf_size];
  int cx = 0;

  if (!util::tty_mode() && last_action > -1) {
    std::string s(2 * last_action + 1, ' ');
    cx += snprintf(buffer + cx, buf_size - cx, "%sx\n", s.c_str());
  }

  column_t blink_column = last_action;
  row_t blink_row = -1;
  if (blink_column >= 0) {
    blink_row = std::countr_one(state.full_mask >> (blink_column * 8)) - 1;
  }
  for (row_t row = kNumRows - 1; row >= 0; --row) {
    cx += print_row(buffer + cx, buf_size - cx, state, row, row == blink_row ? blink_column : -1);
  }
  cx += snprintf(buffer + cx, buf_size - cx, "|1|2|3|4|5|6|7|\n\n");
  if (player_names) {
    cx += snprintf(buffer + cx, buf_size - cx, "%s%s%s: %s\n", ansi::kRed(""), ansi::kCircle("R"),
                   ansi::kReset(""), (*player_names)[kRed].c_str());
    cx += snprintf(buffer + cx, buf_size - cx, "%s%s%s: %s\n\n", ansi::kYellow(""),
                   ansi::kCircle("Y"), ansi::kReset(""), (*player_names)[kYellow].c_str());
  }

  util::release_assert(cx < buf_size, "Buffer overflow (%d < %d)", cx, buf_size);
  ss << buffer << std::endl;
}

void Game::IO::print_mcts_results(std::ostream& ss, const Types::Policy& action_policy,
                                  const Types::SearchResults& results) {
  const auto& valid_actions = std::get<0>(results.valid_actions);
  const auto& action_subpolicy = std::get<0>(action_policy);
  const auto& mcts_counts = std::get<0>(results.counts);
  const auto& net_policy = std::get<0>(results.policy_prior);
  const auto& win_rates = results.win_rates;
  const auto& net_value = results.value_prior;

  constexpr int buf_size = 4096;
  char buffer[buf_size];
  int cx = 0;

  cx += snprintf(buffer + cx, buf_size - cx, "%8s %6s%s%s%s %6s%s%s%s\n", "", "", ansi::kRed(""),
                 ansi::kCircle("R"), ansi::kReset(""), "", ansi::kYellow(""), ansi::kCircle("Y"),
                 ansi::kReset(""));
  cx += snprintf(buffer + cx, buf_size - cx, "%8s %6.3f%% %6.3f%%\n", "net(W)", 100 * net_value(0),
                 100 * net_value(1));
  cx += snprintf(buffer + cx, buf_size - cx, "%8s %6.3f%% %6.3f%%\n", "net(D)", 100 * net_value(2),
                 100 * net_value(2));
  cx += snprintf(buffer + cx, buf_size - cx, "%8s %6.3f%% %6.3f%%\n", "net(L)", 100 * net_value(1),
                 100 * net_value(0));
  cx += snprintf(buffer + cx, buf_size - cx, "%8s %6.3f%% %6.3f%%\n", "win-rate",
                 100 * win_rates(0), 100 * win_rates(1));
  cx += snprintf(buffer + cx, buf_size - cx, "\n");

  cx += snprintf(buffer + cx, buf_size - cx, "%3s %8s %8s %8s\n", "Col", "Net", "Count", "Action");

  for (int i = 0; i < c4::kNumColumns; ++i) {
    if (valid_actions[i]) {
      cx += snprintf(buffer + cx, buf_size - cx, "%3d %8.3f %8.3f %8.3f\n", i + 1, net_policy(i),
                     mcts_counts(i), action_subpolicy(i));
    } else {
      cx += snprintf(buffer + cx, buf_size - cx, "%3d\n", i + 1);
    }
  }

  util::release_assert(cx < buf_size, "Buffer overflow (%d < %d)", cx, buf_size);
  ss << buffer << std::endl;
}

int Game::IO::print_row(char* buf, int n, const State& state, row_t row, column_t blink_column) {
  core::seat_index_t current_player = Rules::get_current_player(state);
  const char* cur_color = current_player == kRed ? ansi::kRed("R") : ansi::kYellow("Y");
  const char* opp_color = current_player == kRed ? ansi::kYellow("Y") : ansi::kRed("R");

  int cx = 0;

  for (int col = 0; col < kNumColumns; ++col) {
    int index = _to_bit_index(row, col);
    bool occupied = (1UL << index) & state.full_mask;
    bool occupied_by_cur_player = (1UL << index) & state.cur_player_mask;

    const char* color = occupied ? (occupied_by_cur_player ? cur_color : opp_color) : "";
    const char* c = occupied ? ansi::kCircle("") : " ";

    cx += snprintf(buf + cx, n - cx, "|%s%s%s%s", col == blink_column ? ansi::kBlink("") : "",
                   color, c, occupied ? ansi::kReset("") : "");
  }

  cx += snprintf(buf + cx, n - cx, "|\n");
  return cx;
}

}  // namespace c4
