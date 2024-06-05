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

inline core::seat_index_t GameState::Data::get_current_player() const {
  return std::popcount(full_mask) % 2;
}

inline core::seat_index_t GameState::Data::get_player_at(int row, int col) const {
  int cp = get_current_player();
  int index = _to_bit_index(row, col);
  bool occupied_by_cur_player = (mask_t(1) << index) & cur_player_mask;
  bool occupied_by_any_player = (mask_t(1) << index) & full_mask;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

inline void GameState::Reflect::apply(Data& data) {
  data.full_mask = __builtin_bswap64(data.full_mask) >> 8;
  data.cur_player_mask = __builtin_bswap64(data.cur_player_mask) >> 8;
}

inline void GameState::Reflect::apply(PolicyTensor& t) {
  PolicyTensor u = eigen_util::reverse(t, t.rank() - 1);
  t = u;
}

inline GameState::SymmetryIndexSet GameState::get_symmetry_indices() const {
  SymmetryIndexSet set;
  set.set();
  return set;
}

inline std::size_t GameState::hash() const {
  return boost::hash_range(&data_.full_mask, &data_.full_mask + 2);
}

inline GameState::ActionMask GameState::get_valid_actions() const {
  mask_t bottomed_full_mask = data_.full_mask + _full_bottom_mask();

  ActionMask mask;
  for (int col = 0; col < kNumColumns; ++col) {
    bool legal = bottomed_full_mask & _column_mask(col);
    mask[col] = legal;
  }
  return mask;
}

inline int GameState::get_move_number() const { return 1 + std::popcount(data_.full_mask); }

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
