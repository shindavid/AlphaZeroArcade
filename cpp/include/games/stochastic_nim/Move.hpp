#pragma once

#include "core/BasicTypes.hpp"
#include "core/PhasedBitSetMoveList.hpp"
#include "games/stochastic_nim/Constants.hpp"

#include <cstdint>
#include <format>
#include <string>

namespace stochastic_nim {

struct GameState;  // forward declaration

// (index, phase) == (-1, 0) will represent an invalid move
class Move {
 public:
  Move() = default;
  Move(int16_t index, core::game_phase_t phase) : index_(index), phase_(phase) {}
  static Move pass() { return Move(kNumMoves, 0); }

  auto operator<=>(const Move&) const = default;

  std::string to_str() const;
  static Move from_str(const GameState&, std::string_view s);

  int16_t index() const { return index_; }
  core::game_phase_t phase() const { return phase_; }

 private:
  int16_t index_;
  core::game_phase_t phase_;
};

using MoveSet = core::PhasedBitSetMoveList<Move, kNumMoves>;

}  // namespace stochastic_nim

template <>
struct std::formatter<stochastic_nim::Move> : std::formatter<std::string> {
  auto format(const stochastic_nim::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(move.to_str(), ctx);
  }
};

#include "inline/games/stochastic_nim/Move.inl"
