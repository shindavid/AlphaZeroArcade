#include "games/connect4/InputFrame.hpp"

#include <bit>

namespace c4 {

inline core::seat_index_t InputFrame::get_player_at(row_t row, column_t col) const {
  int cp = get_current_player();
  int index = to_bit_index(row, col);
  bool occupied_by_cur_player = (mask_t(1) << index) & cur_player_mask;
  bool occupied_by_any_player = (mask_t(1) << index) & full_mask;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

inline core::seat_index_t InputFrame::get_current_player() const {
  return std::popcount(full_mask) % 2;
}

inline int InputFrame::num_empty_cells(column_t col) const {
  return kNumRows - std::popcount(full_mask & column_mask(col));
}

inline constexpr int InputFrame::to_bit_index(row_t row, column_t col) { return 8 * col + row; }

inline constexpr mask_t InputFrame::column_mask(column_t col) { return 63UL << (8 * col); }

inline constexpr mask_t InputFrame::bottom_mask(column_t col) { return 1UL << (8 * col); }

inline constexpr mask_t InputFrame::full_bottom_mask() {
  mask_t mask = 0;
  for (int col = 0; col < kNumColumns; ++col) {
    mask |= bottom_mask(col);
  }
  return mask;
}

}  // namespace c4
