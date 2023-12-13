#include <games/connect4/GameState.hpp>

#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <util/AnsiCodes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

inline std::size_t std::hash<c4::GameState>::operator()(const c4::GameState& state) const {
  return state.hash();
}

namespace c4 {

inline GameState::SymmetryIndexSet GameState::get_symmetry_indices() const {
  SymmetryIndexSet set;
  set.set();
  return set;
}

inline core::seat_index_t GameState::get_current_player() const {
  return std::popcount(full_mask_) % 2;
}

inline GameState::ActionMask GameState::get_valid_actions() const {
  mask_t bottomed_full_mask = full_mask_ + _full_bottom_mask();

  ActionMask mask;
  for (int col = 0; col < kNumColumns; ++col) {
    bool legal = bottomed_full_mask & _column_mask(col);
    mask[col] = legal;
  }
  return mask;
}

inline int GameState::get_move_number() const { return 1 + std::popcount(full_mask_); }

inline core::seat_index_t GameState::get_player_at(int row, int col) const {
  int cp = get_current_player();
  int index = _to_bit_index(row, col);
  bool occupied_by_cur_player = (mask_t(1) << index) & cur_player_mask_;
  bool occupied_by_any_player = (mask_t(1) << index) & full_mask_;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

inline constexpr int GameState::_to_bit_index(row_t row, column_t col) { return 8 * col + row; }

inline constexpr mask_t GameState::_column_mask(column_t col) { return 63UL << (8 * col); }

inline constexpr mask_t GameState::_bottom_mask(column_t col) { return 1UL << (8 * col); }

inline constexpr mask_t GameState::_full_bottom_mask() {
  mask_t mask = 0;
  for (int col = 0; col < kNumColumns; ++col) {
    mask |= _bottom_mask(col);
  }
  return mask;
}

}  // namespace c4
