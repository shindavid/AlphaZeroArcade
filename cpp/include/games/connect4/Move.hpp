#pragma once

#include "core/BitSetMoveList.hpp"
#include "games/connect4/Constants.hpp"
#include "util/StringUtil.hpp"

#include <format>
#include <string>

namespace c4 {

// (row, col) = (8, 0) will represent the pass move
// (row, col) = (0, -1) will represent an invalid move
//
// The cell B4 corresponds to row=3, col=1
class Move {
 public:
  Move() = default;
  Move(int x) : col_(x) {}

  static Move invalid() { return Move(-1); }

  auto operator<=>(const Move&) const = default;

  operator int() const { return col_; }

  std::string to_str() const { return std::to_string(col_ + 1); }
  static Move from_str(std::string_view s) { return Move(util::atoi(s) - 1); }

  std::string serialize() const { return std::format("{}", int(*this)); }
  static Move deserialize(std::string_view s) { return Move(util::atoi(s) - 1); }

 private:
  column_t col_;
};

using MoveList = core::BitSetMoveList<Move, kNumColumns>;

}  // namespace c4

template <>
struct std::formatter<c4::Move> : std::formatter<std::string> {
  auto format(const c4::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(move.to_str(), ctx);
  }
};
