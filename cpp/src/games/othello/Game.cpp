#include <games/othello/Game.hpp>

#include <algorithm>
#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <util/AnsiCodes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

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

// copied from edax-reversi repo - board_next()
void Game::Rules::apply(StateHistory& history, core::action_t action) {
  State& state = history.extend();

  if (action == kPass) {
    std::swap(state.cur_player_mask, state.opponent_mask);
    state.cur_player = 1 - state.cur_player;
    state.pass_count++;
  } else {
    mask_t flipped = flip[action](state.cur_player_mask, state.opponent_mask);
    mask_t cur_player_mask = state.opponent_mask ^ flipped;

    state.opponent_mask = state.cur_player_mask ^ (flipped | (1ULL << action));
    state.cur_player_mask = cur_player_mask;
    state.cur_player = 1 - state.cur_player;
    state.pass_count = 0;
  }
}

bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                              core::action_t last_action, GameResults::Tensor& outcome) {
  if (state.pass_count == kNumPlayers) {
    outcome = compute_outcome(state);
    return true;
  }

  if ((state.opponent_mask | state.cur_player_mask) == kCompleteBoardMask) {
    outcome = compute_outcome(state);
    return true;
  }

  return false;
}

Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
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

  if (!util::tty_mode() && display_last_action) {
    std::string s(2 * blink_col + 3, ' ');
    cx += snprintf(buffer + cx, buf_size - cx, "%sx\n", s.c_str());
  }
  cx += snprintf(buffer + cx, buf_size - cx, "   A B C D E F G H\n");
  for (int row = 0; row < kBoardDimension; ++row) {
    cx += print_row(buffer + cx, buf_size - cx, state, valid_actions, row,
                    row == blink_row ? blink_col : -1);
  }
  cx += snprintf(buffer + cx, buf_size - cx, "\n");
  int opponent_disc_count = std::popcount(state.opponent_mask);
  int cur_player_disc_count = std::popcount(state.cur_player_mask);

  int black_disc_count = state.cur_player == kBlack ? cur_player_disc_count : opponent_disc_count;
  int white_disc_count = state.cur_player == kWhite ? cur_player_disc_count : opponent_disc_count;

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

  util::release_assert(cx < buf_size, "Buffer overflow (%d < %d)", cx, buf_size);
  ss << buffer << std::endl;
}

int Game::IO::print_row(char* buf, int n, const State& state,
                        const Types::ActionMask& valid_actions, row_t row, column_t blink_column) {
  core::seat_index_t current_player = Rules::get_current_player(state);
  const char* cur_color = current_player == kBlack ? ansi::kBlue("*") : ansi::kWhite("0");
  const char* opp_color = current_player == kBlack ? ansi::kWhite("0") : ansi::kBlue("*");

  int cx = 0;

  char prefix = ' ';
  if (!util::tty_mode() && blink_column >= 0) {
    prefix = 'x';
  }
  cx += snprintf(buf + cx, n - cx, "%c%d", prefix, (int)(row + 1));
  for (int col = 0; col < kBoardDimension; ++col) {
    int index = row * kBoardDimension + col;
    bool valid = valid_actions[index];
    bool occupied_by_cur_player = (1UL << index) & state.cur_player_mask;
    bool occupied_by_opp_player = (1UL << index) & state.opponent_mask;
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
  int opponent_count = std::popcount(state.opponent_mask);
  int cur_player_count = std::popcount(state.cur_player_mask);
  if (cur_player_count > opponent_count) {
    return core::WinLossDrawResults::win(state.cur_player);
  } else if (opponent_count > cur_player_count) {
    return core::WinLossDrawResults::win(1 - state.cur_player);
  } else {
    return core::WinLossDrawResults::draw();
  }
}

std::string Game::IO::player_to_str(core::seat_index_t player) {
  return (player == othello::kBlack)
             ? util::create_string("%s%s%s", ansi::kBlue(""), ansi::kCircle("*"), ansi::kReset(""))
             : util::create_string("%s%s%s", ansi::kWhite(""), ansi::kCircle("0"),
                                   ansi::kReset(""));
}

}  // namespace othello
