#pragma once

#include "core/BasicTypes.hpp"
#include "core/PhasedBitSetMoveList.hpp"
#include "games/blokus/Constants.hpp"

#include <cstdint>
#include <format>
#include <string>

namespace blokus {

class GameState;  // forward declaration

// (index, phase) == (-1, 0) will represent an invalid move
class Move {
 public:
  Move() = default;
  Move(int16_t index, core::game_phase_t phase) : index_(index), phase_(phase) {}
  static Move pass() { return Move(kBoardDimension, 0); }

  auto operator<=>(const Move&) const = default;

  bool is_pass() const { return *this == pass(); }
  int16_t index() const { return index_; }
  core::game_phase_t phase() const { return phase_; }

 private:
  int16_t index_;
  core::game_phase_t phase_;
};

using MoveList = core::PhasedBitSetMoveList<Move, kNumMoves>;

}  // namespace blokus
