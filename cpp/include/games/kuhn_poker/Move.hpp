#pragma once

#include "core/PhasedBitSetMoveList.hpp"
#include "games/kuhn_poker/Constants.hpp"
#include "util/StringUtil.hpp"

#include <cstdint>
#include <format>
#include <string>

namespace kuhn_poker {

struct InfoSetState;  // forward declaration

class Move {
 public:
  Move() = default;
  Move(int16_t index, int16_t phase) : index_(index), phase_(phase) {}

  auto operator<=>(const Move&) const = default;

  std::string to_str() const;
  static Move from_str(const InfoSetState&, std::string_view s);

  int16_t index() const { return index_; }
  int16_t phase() const { return phase_; }

 private:
  int16_t index_;
  int16_t phase_;
};

// Max distinct indices in any phase: 6 (deal phase has 6 deals)
using MoveSet = core::PhasedBitSetMoveList<Move, kNumDeals>;

}  // namespace kuhn_poker

template <>
struct std::formatter<kuhn_poker::Move> : std::formatter<std::string> {
  auto format(const kuhn_poker::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(move.to_str(), ctx);
  }
};

#include "inline/games/kuhn_poker/Move.inl"
