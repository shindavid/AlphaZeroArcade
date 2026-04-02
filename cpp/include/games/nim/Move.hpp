#pragma once

#include "core/BitSetMoveList.hpp"
#include "games/nim/Constants.hpp"
#include "util/StringUtil.hpp"

#include <cstdint>
#include <format>
#include <string>

namespace nim {

struct GameState;  // forward declaration

class Move {
 public:
  Move() = default;
  Move(int x) : num_stones_to_take_(x) {}

  auto operator<=>(const Move&) const = default;
  operator int() const { return num_stones_to_take_; }

  int to_json_value() const { return num_stones_to_take_; }
  std::string to_str() const { return std::to_string(num_stones_to_take_ + 1); }
  static Move from_str(const GameState&, std::string_view s) { return Move(util::atoi(s) - 1); }
  std::string serialize() const { return std::format("{}", int(*this)); }
  static Move deserialize(std::string_view s) { return Move(util::atoi(s) - 1); }

 private:
  int8_t num_stones_to_take_;
};

using MoveList = core::BitSetMoveList<Move, kMaxStonesToTake>;

}  // namespace nim

template <>
struct std::formatter<nim::Move> : std::formatter<std::string> {
  auto format(const nim::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(move.to_str(), ctx);
  }
};
