#pragma once

#include "core/BitSetMoveList.hpp"
#include "games/connect4/Constants.hpp"
#include "util/StringUtil.hpp"

#include <format>
#include <string>

namespace c4 {

class GameState;  // forward declaration

// (row, col) = (8, 0) will represent the pass move
// (row, col) = (0, -1) will represent an invalid move
//
// The cell B4 corresponds to row=3, col=1
class Move {
 public:
  Move() = default;
  Move(int x) : col_(x) {}

  auto operator<=>(const Move&) const = default;
  operator int() const { return col_; }

 private:
  column_t col_;
};

using MoveList = core::BitSetMoveList<Move, kNumColumns>;

}  // namespace c4
