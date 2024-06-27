#include <games/connect4/Game.hpp>

#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <util/AnsiCodes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

namespace c4 {

inline size_t Game::BaseState::hash() const {
  return boost::hash_range(&full_mask, &full_mask + 2);
}

inline void Game::Symmetries::apply(BaseState& state, const core::symmetry_t& sym) {
  util::release_assert(sym.group_id == 0, "Unknown group: %d", sym.group_id);
  switch (sym.element) {
    case groups::D1::kIdentity: return;
    case groups::D1::kFlip: {
      state.full_mask = __builtin_bswap64(state.full_mask);
      state.cur_player_mask = __builtin_bswap64(state.cur_player_mask);
      return;
    }
    default: {
      throw util::Exception("Unknown group element: %d", sym.element);
    }
  }
}

inline void Game::Symmetries::apply(Types::PolicyTensor& t, const core::symmetry_t& sym) {
  util::release_assert(sym.group_id == 0, "Unknown group: %d", sym.group_id);
  switch (sym.element) {
    case groups::D1::kIdentity:
      return;
    case groups::D1::kFlip: {
      Types::PolicyTensor u = eigen_util::reverse(t, t.rank() - 1);
      t = u;
      return;
    }
    default: {
      throw util::Exception("Unknown group element: %d", sym.element);
    }
  }
}

inline void Game::Rules::init_state(FullState& state) {
  state.full_mask = 0;
  state.cur_player_mask = 0;
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const FullState& state) {
  const BaseState& base = state;
  mask_t bottomed_full_mask = base.full_mask + _full_bottom_mask();

  Types::ActionMask mask;
  for (int col = 0; col < kNumColumns; ++col) {
    bool legal = bottomed_full_mask & _column_mask(col);
    mask[col] = legal;
  }
  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const BaseState& base) {
  return std::popcount(base.full_mask) % 2;
}

inline Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(const BaseState* start,
                                                                      const BaseState* cur) {
  core::seat_index_t cp = Rules::get_current_player(*cur);
  Tensor tensor;
  for (int row = 0; row < kNumRows; ++row) {
    for (int col = 0; col < kNumColumns; ++col) {
      core::seat_index_t p = _get_player_at(*cur, row, col);
      tensor(0, row, col) = (p == cp);
      tensor(1, row, col) = (p == 1 - cp);
    }
  }
  return tensor;
}

inline Game::TrainingTargets::OwnershipTarget::Tensor
Game::TrainingTargets::OwnershipTarget::tensorize(const Types::GameLogView& view) {
  Tensor tensor;
  core::seat_index_t cp = Rules::get_current_player(*view.cur_pos);
  for (int row = 0; row < kNumRows; ++row) {
    for (int col = 0; col < kNumColumns; ++col) {
      core::seat_index_t p = _get_player_at(*view.final_pos, row, col);
      int val = (p == -1) ? 0 : ((p == cp) ? 2 : 1);
      tensor(row, col) = val;
    }
  }
  return tensor;
}

inline core::seat_index_t Game::_get_player_at(const BaseState& state, row_t row, column_t col) {
  int cp = Rules::get_current_player(state);
  int index = _to_bit_index(row, col);
  bool occupied_by_cur_player = (mask_t(1) << index) & state.cur_player_mask;
  bool occupied_by_any_player = (mask_t(1) << index) & state.full_mask;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

inline constexpr int Game::_to_bit_index(row_t row, column_t col) { return 8 * col + row; }

inline constexpr mask_t Game::_column_mask(column_t col) { return 63UL << (8 * col); }

inline constexpr mask_t Game::_bottom_mask(column_t col) { return 1UL << (8 * col); }

inline constexpr mask_t Game::_full_bottom_mask() {
  mask_t mask = 0;
  for (int col = 0; col < kNumColumns; ++col) {
    mask |= _bottom_mask(col);
  }
  return mask;
}

}  // namespace c4
