#pragma once

#include "core/BitSetMoveList.hpp"
#include "games/tictactoe/Constants.hpp"
#include "util/StringUtil.hpp"

#include <format>
#include <string>

namespace tictactoe {

struct GameState;  // forward declaration

class Move {
 public:
  Move() = default;
  Move(int x) : index_(x) {}

  auto operator<=>(const Move&) const = default;
  operator int() const { return index_; }

  std::string to_str() const { return std::to_string(index_); }
  static Move from_str(const GameState&, std::string_view s) { return Move(util::atoi(s)); }

 private:
  int8_t index_;
};

using MoveSet = core::BitSetMoveList<Move, kNumCells>;

}  // namespace tictactoe

template <>
struct std::formatter<tictactoe::Move> : std::formatter<std::string> {
  auto format(const tictactoe::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(move.to_str(), ctx);
  }
};
