//#include <connect4/C4GameLogic.hpp>

#include <bit>
#include <iostream>

namespace c4 {

inline common::player_index_t GameState::get_current_player() const {
  return std::popcount(masks_[kRed]) != std::popcount(masks_[kYellow]);
}

inline GameResult GameState::apply_move(common::action_index_t action) {
  column_t col = action;
  mask_t full_mask = masks_[kRed] | masks_[kYellow];
  mask_t piece_mask = (full_mask + _bottom_mask(col)) & _column_mask(col);
  common::player_index_t current_player = get_current_player();

  masks_[current_player] |= piece_mask;

  bool win = false;

  constexpr mask_t horizontal_block = 1UL + (1UL<<8) + (1UL<<16) + (1UL<<24);
  constexpr mask_t nw_se_diagonal_block = 1UL + (1UL<<7) + (1UL<<14) + (1UL<<21);
  constexpr mask_t sw_ne_diagonal_block = 1UL + (1UL<<9) + (1UL<<18) + (1UL<<27);

  mask_t masks[] = {
      (piece_mask << 1) - (piece_mask >> 3),  // vertical
      piece_mask * horizontal_block,  // horizontal 1
      (piece_mask << 8) * horizontal_block,  // horizontal 2
      (piece_mask << 16) * horizontal_block,  // horizontal 3
      (piece_mask << 24) * horizontal_block,  // horizontal 4
      piece_mask * nw_se_diagonal_block,  // nw-se diagonal 1
      (piece_mask << 7) * nw_se_diagonal_block,  // nw-se diagonal 2
      (piece_mask << 14) * nw_se_diagonal_block,  // nw-se diagonal 3
      (piece_mask << 21) * nw_se_diagonal_block,  // nw-se diagonal 4
      piece_mask * sw_ne_diagonal_block,  // sw-ne diagonal 1
      (piece_mask << 9) * sw_ne_diagonal_block,  // sw-ne diagonal 2
      (piece_mask << 18) * sw_ne_diagonal_block,  // sw-ne diagonal 3
      (piece_mask << 27) * sw_ne_diagonal_block  // sw-ne diagonal 4
  };

  mask_t updated_mask = masks_[current_player];
  for (mask_t mask : masks) {
    // popcount filters out both int overflow and shift-to-zero
    if (((mask & updated_mask) == mask) && std::popcount(mask) == 4) {
      win = true;
      break;
    }
  }

  GameResult result;
  if (win) {
    result(current_player) = 1.0;
  } else if (std::popcount(full_mask) == 41) {
    result(0) = 0.5;
    result(1) = 0.5;
  }

  return result;
}

inline ActionMask GameState::get_valid_actions() const {
  mask_t full_mask = masks_[kRed] | masks_[kYellow];
  mask_t bottomed_full_mask = full_mask + _full_bottom_mask();

  ActionMask mask;
  for (int col = 0; col < kNumColumns; ++col) {
    bool legal = bottomed_full_mask & _column_mask(col);
    mask[col] = legal;
  }
  return mask;
}

inline std::string GameState::compact_repr() const {
  char buffer[kNumCells + 1];

  for (int i = 0; i < kNumCells; ++i) {
    mask_t piece_mask = 1UL << i;
    if (masks_[kRed] & piece_mask) {
      buffer[i] = 'R';
    } else if (masks_[kYellow] & piece_mask) {
      buffer[i] = 'Y';
    } else {
      buffer[i] = '.';
    }
  }
  buffer[kNumCells] = 0;
  return buffer;
}

inline bool GameState::operator==(const GameState& other) const {
  return masks_[kRed] == other.masks_[kRed] && masks_[kYellow] == other.masks_[kYellow];
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
