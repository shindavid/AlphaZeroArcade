#pragma once

#include "chess-library/include/chess.hpp"
#include "games/chess/GameState.hpp"
#include "util/StringUtil.hpp"

#include <algorithm>
#include <format>
#include <random>

namespace a0achess {

class Move : public chess::Move {
 public:
  Move() = default;
  Move(const chess::Move& move) : chess::Move(move) {}

  auto operator<=>(const Move& m) const { return move() <=> m.move(); }

  int to_json_value() const { return move(); }
  std::string to_str() const { return chess::uci::moveToUci(*this); }
  static Move from_str(const GameState& state, std::string_view s);
  std::string serialize() const { return std::format("{}", move()); }
  static Move deserialize(std::string_view s) { return Move(util::atoi(s)); }
};

class MoveList : public chess::Movelist {
 public:
  static constexpr bool kSortedByMove = false;
  using chess::Movelist::Movelist;

  int count() const { return size(); }  // TODO: rename count() to size in the MoveList interface
  Move get_random(std::mt19937& prng) const;  // assumes !empty()
  bool contains(const Move& m) const { return std::find(begin(), end(), m) != end(); }

  size_t serialize(char* buffer) const;
  size_t deserialize(const char* buffer);
  std::string to_string() const;  // should be called to_json()
};

}  // namespace a0achess

template <>
struct std::formatter<a0achess::Move> : std::formatter<std::string> {
  auto format(const a0achess::Move& move, format_context& ctx) const {
    return std::formatter<std::string>::format(move.to_str(), ctx);
  }
};

#include "inline/games/chess/Move.inl"
