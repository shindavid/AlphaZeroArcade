#include "games/othello/Game.hpp"

#include "util/AnsiCodes.hpp"
#include "util/Rendering.hpp"

#include <boost/lexical_cast.hpp>

#include <algorithm>
#include <bit>
#include <format>
#include <iostream>

namespace othello {

std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
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

void Game::Rules::apply(State& state, core::action_t action) {
  if (action == kPass) {
    std::swap(state.core.cur_player_mask, state.core.opponent_mask);
    state.core.cur_player = 1 - state.core.cur_player;
    state.core.pass_count++;
  } else {
    mask_t flipped = flip[action](state.core.cur_player_mask, state.core.opponent_mask);
    mask_t cur_player_mask = state.core.opponent_mask ^ flipped;

    state.core.opponent_mask = state.core.cur_player_mask ^ (flipped | (1ULL << action));
    state.core.cur_player_mask = cur_player_mask;
    state.core.cur_player = 1 - state.core.cur_player;
    state.core.pass_count = 0;
    state.compute_aux();
  }
}

bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                              core::action_t last_action, GameResults::Tensor& outcome) {
  if (state.core.pass_count == kNumPlayers) {
    outcome = compute_outcome(state);
    return true;
  }

  if ((state.core.opponent_mask | state.core.cur_player_mask) == kCompleteBoardMask) {
    outcome = compute_outcome(state);
    return true;
  }

  return false;
}

Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  uint64_t mask = get_moves(state.core.cur_player_mask, state.core.opponent_mask);
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

void Game::IO::print_state(std::ostream& ss, const State& state, core::action_t last_action,
                           const Types::player_name_array_t* player_names) {
  Types::ActionMask valid_actions = Rules::get_legal_moves(state);
  bool display_last_action = last_action >= 0;
  int blink_row = -1;
  int blink_col = -1;
  if (display_last_action && last_action != kPass) {
    blink_row = last_action / kBoardDimension;
    blink_col = last_action % kBoardDimension;
  }

  constexpr int buf_size = 4096;
  char buffer[buf_size];
  int cx = 0;

  if (util::Rendering::mode() == util::Rendering::kText && display_last_action) {
    std::string s(2 * blink_col + 3, ' ');
    cx += snprintf(buffer + cx, buf_size - cx, "%sx\n", s.c_str());
  }
  cx += snprintf(buffer + cx, buf_size - cx, "   A B C D E F G H\n");
  for (int row = 0; row < kBoardDimension; ++row) {
    cx += print_row(buffer + cx, buf_size - cx, state, valid_actions, row,
                    row == blink_row ? blink_col : -1);
  }
  cx += snprintf(buffer + cx, buf_size - cx, "\n");
  int opponent_disc_count = std::popcount(state.core.opponent_mask);
  int cur_player_disc_count = std::popcount(state.core.cur_player_mask);

  int black_disc_count =
    state.core.cur_player == kBlack ? cur_player_disc_count : opponent_disc_count;
  int white_disc_count =
    state.core.cur_player == kWhite ? cur_player_disc_count : opponent_disc_count;

  cx += snprintf(buffer + cx, buf_size - cx, "Score: Player\n");
  cx += snprintf(buffer + cx, buf_size - cx, "%5d: %s%s%s", black_disc_count, ansi::kBlue(""),
                 ansi::kCircle("*"), ansi::kReset(""));
  if (player_names) {
    cx += snprintf(buffer + cx, buf_size - cx, " [%s]", (*player_names)[kBlack].c_str());
  }
  cx += snprintf(buffer + cx, buf_size - cx, "\n");

  cx += snprintf(buffer + cx, buf_size - cx, "%5d: %s%s%s", white_disc_count, ansi::kWhite(""),
                 ansi::kCircle("0"), ansi::kReset(""));
  if (player_names) {
    cx += snprintf(buffer + cx, buf_size - cx, " [%s]", (*player_names)[kWhite].c_str());
  }
  cx += snprintf(buffer + cx, buf_size - cx, "\n");

  RELEASE_ASSERT(cx < buf_size, "Buffer overflow ({} < {})", cx, buf_size);
  ss << buffer << std::endl;
}

void Game::IO::write_edax_board_str(char* buf, const State& state) {
  char chars[3];

  // if cur_player == 0 (X), then chars should be ".XO"
  // if cur_player == 1 (O), then chars should be ".OX"
  chars[0] = '.';
  chars[state.core.cur_player + 1] = 'X';
  chars[2 - state.core.cur_player] = 'O';

  int cx = 0;
  cx += snprintf(buf + cx, 76, "setboard ");
  for (int i = 0; i < 64; ++i) {
    int cur = (state.core.cur_player_mask >> i) & 1;
    int opp = (state.core.opponent_mask >> i) & 1;
    int x = 2 * opp + cur;  // 0 = empty, 1 = cur, 2 = opp
    buf[cx++] = chars[x];
  }
  buf[cx++] = ' ';
  buf[cx++] = (state.core.cur_player == kBlack) ? 'X' : 'O';
  buf[cx++] = '\n';
  DEBUG_ASSERT(cx == 76, "Unexpected error ({} != {})", cx, 76);
}

int Game::IO::print_row(char* buf, int n, const State& state,
                        const Types::ActionMask& valid_actions, row_t row, column_t blink_column) {
  core::seat_index_t current_player = Rules::get_current_player(state);
  const char* cur_color = current_player == kBlack ? ansi::kBlue("*") : ansi::kWhite("0");
  const char* opp_color = current_player == kBlack ? ansi::kWhite("0") : ansi::kBlue("*");

  int cx = 0;

  char prefix = ' ';
  if (util::Rendering::mode() == util::Rendering::kText && blink_column >= 0) {
    prefix = 'x';
  }
  cx += snprintf(buf + cx, n - cx, "%c%d", prefix, (int)(row + 1));
  for (int col = 0; col < kBoardDimension; ++col) {
    int index = row * kBoardDimension + col;
    bool valid = valid_actions[index];
    bool occupied_by_cur_player = (1UL << index) & state.core.cur_player_mask;
    bool occupied_by_opp_player = (1UL << index) & state.core.opponent_mask;
    bool occupied = occupied_by_cur_player || occupied_by_opp_player;

    const char* color =
      occupied_by_cur_player ? cur_color : (occupied_by_opp_player ? opp_color : "");
    const char* c = occupied ? ansi::kCircle("") : (valid ? "." : " ");

    cx += snprintf(buf + cx, n - cx, "|%s%s%s%s", color, c,
                   col == blink_column ? ansi::kBlink("") : "", occupied ? ansi::kReset("") : "");
  }

  cx += snprintf(buf + cx, n - cx, "|%s\n", ansi::kReset(""));
  return cx;
}

Game::GameResults::Tensor Game::Rules::compute_outcome(const State& state) {
  int opponent_count = std::popcount(state.core.opponent_mask);
  int cur_player_count = std::popcount(state.core.cur_player_mask);
  if (cur_player_count > opponent_count) {
    return core::WinLossDrawResults::win(state.core.cur_player);
  } else if (opponent_count > cur_player_count) {
    return core::WinLossDrawResults::win(1 - state.core.cur_player);
  } else {
    return core::WinLossDrawResults::draw();
  }
}

std::string Game::IO::player_to_str(core::seat_index_t player) {
  return (player == othello::kBlack)
           ? std::format("{}{}{}", ansi::kBlue(""), ansi::kCircle("*"), ansi::kReset(""))
           : std::format("{}{}{}", ansi::kWhite(""), ansi::kCircle("0"), ansi::kReset(""));
}

boost::json::value Game::IO::state_to_json(const State& state) {
  char buf[kBoardDimension * kBoardDimension + 1];
  const char* syms = ".*0";

  int c = 0;
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      buf[c++] = syms[state.get_player_at(row, col) + 1];
    }
  }
  buf[c] = '\0';

  return boost::json::value(std::string(buf));
}

}  // namespace othello
