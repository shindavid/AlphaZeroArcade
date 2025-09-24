#include "games/connect4/Game.hpp"

#include "core/DefaultCanonicalizer.hpp"

#include <boost/lexical_cast.hpp>

#include <bit>

namespace c4 {

inline size_t Game::State::hash() const {
  auto tuple = std::make_tuple(full_mask, cur_player_mask);
  std::hash<decltype(tuple)> hasher;
  return hasher(tuple);
}

inline int Game::State::num_empty_cells(column_t col) const {
  return kNumRows - std::popcount(full_mask & _column_mask(col));
}

inline core::seat_index_t Game::State::get_player_at(row_t row, column_t col) const {
  int cp = Rules::get_current_player(*this);
  int index = _to_bit_index(row, col);
  bool occupied_by_cur_player = (mask_t(1) << index) & cur_player_mask;
  bool occupied_by_any_player = (mask_t(1) << index) & full_mask;
  return occupied_by_any_player ? (occupied_by_cur_player ? cp : (1 - cp)) : -1;
}

inline Game::Types::SymmetryMask Game::Symmetries::get_mask(const State& state) {
  Types::SymmetryMask mask;
  mask.set();
  return mask;
}

inline void Game::Symmetries::apply(State& state, group::element_t sym) {
  switch (sym) {
    case groups::D1::kIdentity:
      return;
    case groups::D1::kFlip: {
      state.full_mask = std::byteswap(state.full_mask << 8);
      state.cur_player_mask = std::byteswap(state.cur_player_mask << 8);
      return;
    }
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

inline void Game::Symmetries::apply(Types::PolicyTensor& t, group::element_t sym,
                                    core::action_mode_t) {
  switch (sym) {
    case groups::D1::kIdentity:
      return;
    case groups::D1::kFlip: {
      Types::PolicyTensor u = eigen_util::reverse(t, t.rank() - 1);
      t = u;
      return;
    }
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

inline void Game::Symmetries::apply(core::action_t& action, group::element_t sym,
                                    core::action_mode_t) {
  switch (sym) {
    case groups::D1::kIdentity:
      return;
    case groups::D1::kFlip: {
      action = 6 - action;
      return;
    }
    default: {
      throw util::Exception("Unknown group element: {}", sym);
    }
  }
}

inline group::element_t Game::Symmetries::get_canonical_symmetry(const State& state) {
  using DefaultCanonicalizer = core::DefaultCanonicalizer<Game>;
  return DefaultCanonicalizer::get(state);
}

inline void Game::Rules::init_state(State& state) {
  state.full_mask = 0;
  state.cur_player_mask = 0;
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  mask_t bottomed_full_mask = state.full_mask + _full_bottom_mask();

  Types::ActionMask mask;
  for (int col = 0; col < kNumColumns; ++col) {
    bool legal = bottomed_full_mask & _column_mask(col);
    mask[col] = legal;
  }
  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return std::popcount(state.full_mask) % 2;
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  column_t col = action;
  mask_t piece_mask = (state.full_mask + _bottom_mask(col)) & _column_mask(col);

  state.cur_player_mask ^= state.full_mask;
  state.full_mask |= piece_mask;
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
