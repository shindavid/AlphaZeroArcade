#include "games/connect4/Game.hpp"

#include "util/AnsiCodes.hpp"
#include "util/Rendering.hpp"

#include <boost/lexical_cast.hpp>

#include <bit>
#include <iostream>

namespace c4 {

bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                              core::action_t last_action, GameResults::Tensor& outcome) {
  constexpr mask_t horizontal_block = 1UL + (1UL << 8) + (1UL << 16) + (1UL << 24);
  constexpr mask_t nw_se_diagonal_block = 1UL + (1UL << 7) + (1UL << 14) + (1UL << 21);
  constexpr mask_t sw_ne_diagonal_block = 1UL + (1UL << 9) + (1UL << 18) + (1UL << 27);

  column_t col = last_action;
  mask_t piece_mask = ((state.full_mask + _bottom_mask(col)) & (_column_mask(col) << 1)) >> 1;

  RELEASE_ASSERT(last_player != get_current_player(state), "Wrong player ({} == {})", last_player,
                 get_current_player(state));
  RELEASE_ASSERT(last_action >= 0, "Bad last_action: {}", last_action);
  RELEASE_ASSERT(std::popcount(piece_mask) == 1, "Wrong popcount({})={}", piece_mask,
                 std::popcount(piece_mask));

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

  if (util::Rendering::mode() == util::Rendering::kText && last_action > -1) {
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

  RELEASE_ASSERT(cx < buf_size, "Buffer overflow ({} < {})", cx, buf_size);
  ss << buffer << std::endl;
}

std::string Game::IO::compact_state_repr(const State& state) {
  // 6 lines joined by '\n' of 7 char's each, each char is in {'R', 'Y', '_'}
  core::seat_index_t cp = Rules::get_current_player(state);
  char cur_color = cp == kRed ? 'R' : 'Y';
  char opp_color = cp == kRed ? 'Y' : 'R';

  std::string repr;
  for (int row = kNumRows - 1; row >= 0; --row) {
    for (int col = 0; col < kNumColumns; ++col) {
      int index = _to_bit_index(row, col);
      bool occupied = (1UL << index) & state.full_mask;
      bool occupied_by_cur_player = (1UL << index) & state.cur_player_mask;

      char c = occupied ? (occupied_by_cur_player ? cur_color : opp_color) : '_';
      repr += c;
    }
    repr += '\n';
  }
  return repr;
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

std::string Game::IO::player_to_str(core::seat_index_t player) {
  return (player == c4::kRed)
           ? std::format("{}{}{}", ansi::kRed(""), ansi::kCircle("R"), ansi::kReset(""))
           : std::format("{}{}{}", ansi::kYellow(""), ansi::kCircle("Y"), ansi::kReset(""));
}

}  // namespace c4
