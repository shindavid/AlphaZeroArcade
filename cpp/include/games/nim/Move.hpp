#pragma once

#include "core/BitSetMoveList.hpp"
#include "games/nim/Constants.hpp"

#include <cstdint>

namespace nim {

struct GameState;  // forward declaration

class Move {
 public:
  Move() = default;
  Move(int x) : num_stones_to_take_(x) {}

  auto operator<=>(const Move&) const = default;
  operator int() const { return num_stones_to_take_; }

 private:
  int8_t num_stones_to_take_;
};

using MoveList = core::BitSetMoveList<Move, kMaxStonesToTake>;

}  // namespace nim
