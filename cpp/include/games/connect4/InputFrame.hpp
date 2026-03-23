#pragma once

#include "core/BasicTypes.hpp"
#include "games/connect4/Constants.hpp"
#include "util/CppUtil.hpp"

#include <functional>

namespace c4 {

struct InputFrame {
  auto operator<=>(const InputFrame& other) const = default;
  size_t hash() const { return util::PODHash<InputFrame>{}(*this); }
  core::seat_index_t get_player_at(row_t row, column_t col) const;
  core::seat_index_t get_current_player() const;
  int num_empty_cells(column_t col) const;
  static constexpr int to_bit_index(row_t row, column_t col);
  static constexpr mask_t column_mask(column_t col);
  static constexpr mask_t bottom_mask(column_t col);
  static constexpr mask_t full_bottom_mask();

  mask_t full_mask;        // spaces occupied by either player
  mask_t cur_player_mask;  // spaces occupied by current player
};

}  // namespace c4

namespace std {

template <>
struct hash<c4::InputFrame> {
  size_t operator()(const c4::InputFrame& frame) const { return frame.hash(); }
};

}  // namespace std

#include "inline/games/connect4/InputFrame.inl"
