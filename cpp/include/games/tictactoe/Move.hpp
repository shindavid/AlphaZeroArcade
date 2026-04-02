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

 private:
  int8_t index_;
};

using MoveList = core::BitSetMoveList<Move, kNumCells>;

}  // namespace tictactoe
