#pragma once

#include "core/BitSetMoveList.hpp"
#include "games/hex/Constants.hpp"
#include "games/hex/GameState.hpp"
#include "games/hex/Types.hpp"

#include <cstdint>
#include <format>
#include <string>

namespace hex {

struct GameState;  // forward declaration

// (row, col) = (+11, 0) will represent the swap move
// (row, col) = (0, -1) will represent an invalid move
//
// The cell B4 corresponds to row=3, col=1
class Move {
 public:
  Move() = default;
  Move(int row, int col) : row_(row), col_(col) {}
  Move(vertex_t v) : row_(v / Constants::kBoardDim), col_(v % Constants::kBoardDim) {}
  static Move swap() { return Move(Constants::kBoardDim, 0); }
  Move transpose() const { return Move(col_, row_); }

  auto operator<=>(const Move&) const = default;
  operator int() const { return vertex(); }

  vertex_t vertex() const { return row_ * Constants::kBoardDim + col_; }
  int8_t row() const { return row_; }
  int8_t col() const { return col_; }

 private:
  int8_t row_;
  int8_t col_;
};

using MoveList = core::BitSetMoveList<Move, Constants::kNumMoves>;

}  // namespace hex
