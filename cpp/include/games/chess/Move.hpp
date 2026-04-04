#pragma once

#include "games/chess/GameState.hpp"

#include <chess-library/include/chess.hpp>

#include <algorithm>
#include <format>
#include <random>
#include <string>

namespace a0achess {

class Move : public chess::Move {
 public:
  Move() = default;
  Move(const chess::Move& move) : chess::Move(move) {}

  auto operator<=>(const Move& m) const { return move() <=> m.move(); }

  std::string to_str() const { return chess::uci::moveToUci(*this); }
  static Move from_str(const GameState& state, std::string_view s);
};

class MoveList : public chess::Movelist {
 public:
  static constexpr bool kSortedByMove = false;
  using chess::Movelist::Movelist;

  int count() const { return size(); }  // TODO: rename count() to size in the MoveList interface
  Move get_random(std::mt19937& prng) const;  // assumes !empty()
  bool contains(const Move& m) const { return std::find(begin(), end(), m) != end(); }

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
