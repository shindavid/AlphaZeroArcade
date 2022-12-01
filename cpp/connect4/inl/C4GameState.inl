#include <connect4/C4GameState.hpp>

#include <util/AnsiCodes.hpp>
#include <util/PrintUtil.hpp>

#include <bit>
#include <iostream>

namespace c4 {

inline common::player_index_t GameState::get_current_player() const {
  return std::popcount(full_mask_) % 2;
}

inline common::GameStateTypes<GameState>::Result GameState::apply_move(common::action_index_t action) {
  column_t col = action;
  mask_t piece_mask = (full_mask_ + _bottom_mask(col)) & _column_mask(col);
  common::player_index_t current_player = get_current_player();

  cur_player_mask_ ^= full_mask_;
  full_mask_ |= piece_mask;

  bool win = false;

  constexpr mask_t horizontal_block = 1UL + (1UL<<8) + (1UL<<16) + (1UL<<24);
  constexpr mask_t nw_se_diagonal_block = 1UL + (1UL<<7) + (1UL<<14) + (1UL<<21);
  constexpr mask_t sw_ne_diagonal_block = 1UL + (1UL<<9) + (1UL<<18) + (1UL<<27);

  mask_t masks[] = {
      (piece_mask << 1) - (piece_mask >> 3),  // vertical
      piece_mask * horizontal_block,  // horizontal 1
      (piece_mask >> 8) * horizontal_block,  // horizontal 2
      (piece_mask >> 16) * horizontal_block,  // horizontal 3
      (piece_mask >> 24) * horizontal_block,  // horizontal 4
      piece_mask * nw_se_diagonal_block,  // nw-se diagonal 1
      (piece_mask >> 7) * nw_se_diagonal_block,  // nw-se diagonal 2
      (piece_mask >> 14) * nw_se_diagonal_block,  // nw-se diagonal 3
      (piece_mask >> 21) * nw_se_diagonal_block,  // nw-se diagonal 4
      piece_mask * sw_ne_diagonal_block,  // sw-ne diagonal 1
      (piece_mask >> 9) * sw_ne_diagonal_block,  // sw-ne diagonal 2
      (piece_mask >> 18) * sw_ne_diagonal_block,  // sw-ne diagonal 3
      (piece_mask >> 27) * sw_ne_diagonal_block  // sw-ne diagonal 4
  };

  mask_t updated_mask = full_mask_ ^ cur_player_mask_;
  for (mask_t mask : masks) {
    // popcount filters out both int overflow and shift-to-zero
    if (((mask & updated_mask) == mask) && std::popcount(mask) == 4) {
      win = true;
      break;
    }
  }

  Result result;
  result.setZero();
  if (win) {
    result(current_player) = 1.0;
  } else if (std::popcount(full_mask_) == kNumCells) {
    result(0) = 0.5;
    result(1) = 0.5;
  }

  return result;
}

inline ActionMask GameState::get_valid_actions() const {
  mask_t bottomed_full_mask = full_mask_ + _full_bottom_mask();

  ActionMask mask;
  for (int col = 0; col < kNumColumns; ++col) {
    bool legal = bottomed_full_mask & _column_mask(col);
    mask[col] = legal;
  }
  return mask;
}

inline std::string GameState::compact_repr() const {
  char buffer[kNumCells + 1];

  common::player_index_t current_player = get_current_player();
  char cur_color = current_player == kRed ? 'R' : 'Y';
  char opp_color = current_player == kRed ? 'Y' : 'R';

  for (int col = 0; col < kNumColumns; ++col) {
    for (int row = 0; row < kNumRows; ++row) {
      int read_index = _to_bit_index(col, row);
      mask_t piece_mask = 1UL << read_index;
      int write_index = 6 * col + row;
      if (cur_player_mask_ & piece_mask) {
        buffer[write_index] = cur_color;
      } else if (full_mask_ & piece_mask) {
        buffer[write_index] = opp_color;
      } else {
        buffer[write_index] = '.';
      }
    }
  }
  buffer[kNumCells] = 0;
  return buffer;
}


template<eigen_util::FixedTensorConcept InputTensor> void GameState::tensorize(int slice, InputTensor& tensor) const {
  mask_t opp_player_mask = full_mask_ ^ cur_player_mask_;
  for (int col = 0; col < kNumColumns; ++col) {
    for (int row = 0; row < kNumRows; ++row) {
      int index = _to_bit_index(col, row);
      bool occupied_by_cur_player = (1UL << index) & cur_player_mask_;
      tensor(slice, 0, col, row) = occupied_by_cur_player;
    }
  }
  for (int col = 0; col < kNumColumns; ++col) {
    for (int row = 0; row < kNumRows; ++row) {
      int index = _to_bit_index(col, row);
      bool occupied_by_opp_player = (1UL << index) & opp_player_mask;
      tensor(slice, 1, col, row) = occupied_by_opp_player;
    }
  }
}

inline void GameState::xprintf_dump(const player_name_array_t& player_names, common::action_index_t last_action) const {
  column_t blink_column = last_action;
  row_t blink_row = -1;
  if (blink_column >= 0) {
    blink_row = std::countr_one(full_mask_ >> (blink_column * 8)) - 1;
  }
  for (row_t row = kNumRows - 1; row >= 0; --row) {
    xprintf_row_dump(row, row == blink_row ? blink_column : -1);
  }
  util::xprintf("|1|2|3|4|5|6|7|\n");
  util::xprintf("%s%s%s: %s\n", ansi::kRed, ansi::kCircle, ansi::kReset, player_names[kRed].c_str());
  util::xprintf("%s%s%s: %s\n\n", ansi::kYellow, ansi::kCircle, ansi::kReset, player_names[kYellow].c_str());
  util::xflush();
}

inline void GameState::xprintf_row_dump(row_t row, column_t blink_column) const {
  common::player_index_t current_player = get_current_player();
  const char* cur_color = current_player == kRed ? ansi::kRed : ansi::kYellow;
  const char* opp_color = current_player == kRed ? ansi::kYellow : ansi::kRed;

  for (int col = 0; col < kNumColumns; ++col) {
    int index = _to_bit_index(col, row);
    bool occupied = (1UL << index) & full_mask_;
    bool occupied_by_cur_player = (1UL << index) & cur_player_mask_;

    const char* color = occupied ? (occupied_by_cur_player ? cur_color : opp_color) : "";
    const char* c = occupied ? ansi::kCircle : " ";

    util::xprintf("|%s%s%s%s", col == blink_column ? ansi::kBlink : "", color, c, occupied ? ansi::kReset : "");
  }

  util::xprintf("|\n");
}

inline bool GameState::operator==(const GameState& other) const {
  return full_mask_ == other.full_mask_ && cur_player_mask_ == other.cur_player_mask_;
}

inline constexpr int GameState::_to_bit_index(column_t col, row_t row) {
  return 8 * col + row;
}

inline constexpr mask_t GameState::_column_mask(column_t col) {
  return 63UL << (8 * col);
}

inline constexpr mask_t GameState::_bottom_mask(column_t col) {
  return 1UL << (8 * col);
}

inline constexpr mask_t GameState::_full_bottom_mask() {
  mask_t mask = 0;
  for (int col = 0; col < kNumColumns; ++col) {
    mask |= _bottom_mask(col);
  }
  return mask;
}

}  // namespace c4
