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

  std::string to_str() const;
  static Move from_str(const GameState&, std::string_view s);

 private:
  int8_t row_;
  int8_t col_;
};

using MoveSet = core::BitSetMoveList<Move, Constants::kNumMoves>;

}  // namespace hex

template <>
struct std::formatter<hex::Move> : std::formatter<std::string> {
  auto format(const hex::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(move.to_str(), ctx);
  }
};

#include "inline/games/hex/Move.inl"
