#pragma once

#include "core/BasicTypes.hpp"
#include "core/PhasedBitSetMoveList.hpp"
#include "games/blokus/Constants.hpp"

#include <cstdint>
#include <format>
#include <string>

namespace blokus {

// (index, phase) == (-1, 0) will represent an invalid move
class Move {
 public:
  Move() = default;
  Move(int16_t index, core::game_phase_t phase) : index_(index), phase_(phase) {}

  static Move invalid() { return Move(-1, 0); }
  static Move pass() { return Move(kBoardDimension, 0); }

  auto operator<=>(const Move&) const = default;

  bool is_pass() const { return *this == pass(); }
  int16_t index() const { return index_; }
  core::game_phase_t phase() const { return phase_; }

  std::string to_str() const;
  static Move from_str(std::string_view s);

  std::string serialize() const;
  static Move deserialize(std::string_view s);

 private:
  int16_t index_;
  core::game_phase_t phase_;
};

using MoveList = core::PhasedBitSetMoveList<Move, kNumMoves>;

}  // namespace blokus

template <>
struct std::formatter<blokus::Move> : std::formatter<std::string> {
  auto format(const blokus::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(move.to_str(), ctx);
  }
};

#include "inline/games/blokus/Move.inl"
