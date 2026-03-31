#pragma once

#include "core/BitSetMoveList.hpp"
#include "games/othello/Constants.hpp"

#include <cstdint>
#include <format>
#include <string>

namespace othello {

// (row, col) = (8, 0) will represent the pass move
// (row, col) = (0, -1) will represent an invalid move
//
// The cell B4 corresponds to row=3, col=1
class Move {
 public:
  Move() = default;
  Move(int row, int col) : row_(row), col_(col) {}
  Move(int x) : row_(x / kBoardDimension), col_(x % kBoardDimension) {}

  static Move invalid() { return Move(0, -1); }
  static Move pass() { return Move(kBoardDimension, 0); }

  auto operator<=>(const Move&) const = default;

  operator int() const { return row_ * kBoardDimension + col_; }

  int8_t row() const { return row_; }
  int8_t col() const { return col_; }

  std::string to_str() const;
  static Move from_str(std::string_view s);

  std::string serialize() const;
  static Move deserialize(std::string_view s);

  Move transpose() const { return Move(col_, row_); }

 private:
  int8_t row_;
  int8_t col_;
};

using MoveList = core::BitSetMoveList<Move, kNumGlobalActions>;

}  // namespace othello

template <>
struct std::formatter<othello::Move> : std::formatter<std::string> {
  auto format(const othello::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(move.to_str(), ctx);
  }
};

#include "inline/games/othello/Move.inl"
