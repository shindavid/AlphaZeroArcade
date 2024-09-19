#include <games/connect4/Game.hpp>

#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <core/DefaultCanonicalizer.hpp>
#include <util/AnsiCodes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

namespace c4 {

inline size_t Game::BaseState::hash() const {
  auto tuple = std::make_tuple(full_mask, cur_player_mask);
  std::hash<decltype(tuple)> hasher;
  return hasher(tuple);
}

inline Game::Types::SymmetryMask Game::Symmetries::get_mask(const BaseState& state) {
  Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline void Game::Symmetries::apply(BaseState& state, group::element_t sym) {
  switch (sym) {
    case groups::D1::kIdentity: return;
    case groups::D1::kFlip: {
      state.full_mask = __builtin_bswap64(state.full_mask << 8);
      state.cur_player_mask = __builtin_bswap64(state.cur_player_mask << 8);
      return;
    }
    default: {
      throw util::Exception("Unknown group element: %d", sym);
    }
  }
}

inline void Game::Symmetries::apply(Types::PolicyTensor& t, group::element_t sym) {
  switch (sym) {
    case groups::D1::kIdentity:
      return;
    case groups::D1::kFlip: {
      Types::PolicyTensor u = eigen_util::reverse(t, t.rank() - 1);
      t = u;
      return;
    }
    default: {
      throw util::Exception("Unknown group element: %d", sym);
    }
  }
}

inline void Game::Symmetries::apply(core::action_t& action, group::element_t sym) {
  switch (sym) {
    case groups::D1::kIdentity: return;
    case groups::D1::kFlip: {
      action = 6 - action;
      return;
    }
    default: {
      throw util::Exception("Unknown group element: %d", sym);
    }
  }
}

inline group::element_t Game::Symmetries::get_canonical_symmetry(const BaseState& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<Game>;
  return DefaultCanonicalizer::get(state);
}

inline void Game::Rules::init_state(FullState& state, group::element_t sym) {
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
  tensor.setZero();
  int i = 0;
  const BaseState* state = cur;
  while (true) {
    for (int row = 0; row < kNumRows; ++row) {
      for (int col = 0; col < kNumColumns; ++col) {
        core::seat_index_t p = _get_player_at(*state, row, col);
        if (p < 0) continue;
        int x = (Constants::kNumPlayers + cp - p) % Constants::kNumPlayers;
        tensor(i + x, row, col) = 1;
      }
    }
    if (state == start) break;
    state--;
    i += kNumPlayers;
  }
  return tensor;
}

inline Game::TrainingTargets::OwnershipTarget::Tensor
Game::TrainingTargets::OwnershipTarget::tensorize(const Types::GameLogView& view) {
  Tensor tensor;
  tensor.setZero();
  core::seat_index_t cp = Rules::get_current_player(*view.cur_pos);
  for (int row = 0; row < kNumRows; ++row) {
    for (int col = 0; col < kNumColumns; ++col) {
      core::seat_index_t p = _get_player_at(*view.final_pos, row, col);
      int x = (p == -1) ? 2 : ((p == cp) ? 0 : 1);
      tensor(x, row, col) = 1;
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
