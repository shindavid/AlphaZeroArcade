#pragma once

#include <cstdint>
#include <functional>

#include <boost/functional/hash.hpp>

#include <connect4/Constants.hpp>

namespace c4 {

/*
 * Bit order encoding for the board:
 *
 * 12 19 26 33 40 47 54
 * 11 18 25 32 39 46 53
 * 10 17 24 31 38 45 52
 *  9 16 23 30 37 44 51
 *  8 15 22 29 36 43 50
 *  7 14 21 28 35 42 49
 *
 *  NOT:
 *
 *  5 13 21 29 37 45 53
 *  4 12 20 28 36 44 52
 *  3 11 19 27 35 43 51
 *  2 10 18 26 34 42 50
 *  1  9 17 25 33 41 49
 *  0  8 16 24 32 40 48
 */
class GameState {
public:
  static int get_num_global_actions() { return kNumColumns; }
  common::player_index_t get_current_player() const;
  GameResult apply_move(common::action_index_t action);
  ActionMask get_valid_actions() const;
  std::string compact_repr() const;

  bool operator==(const GameState& other) const;
  std::size_t hash() const { return boost::hash_range(&full_mask_, (&full_mask_) + 2); }

private:
  static constexpr mask_t _column_mask(column_t col);  // mask containing piece on all cells of given column
  static constexpr mask_t _bottom_mask(column_t col);  // mask containing single piece at bottom cell
  static constexpr mask_t _full_bottom_mask();  // mask containing piece in each bottom cell

  mask_t full_mask_ = 0;  // spaces occupied by either player
  mask_t cur_player_mask_ = 0;  // spaces occupied by current player
  //mask_t masks_[kNumPlayers] = {};
};

}  // namespace c4

template <>
struct std::hash<c4::GameState> {
  std::size_t operator()(const c4::GameState& state) const { return state.hash(); }
};

namespace c4 {
static_assert(common::AbstractGameState<GameState>);
}  // namespace c4

#include <connect4/C4GameLogicINLINES.cpp>

