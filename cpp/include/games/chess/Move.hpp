#pragma once

#include "games/chess/GameState.hpp"
#include "util/StringUtil.hpp"

#include <chess-library/include/chess.hpp>

#include <algorithm>
#include <format>
#include <random>

namespace a0achess {

class Move : public chess::Move {
 public:
  Move() = default;
  Move(const chess::Move& move) : chess::Move(move) {}

  auto operator<=>(const Move& m) const { return move() <=> m.move(); }
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

#include "inline/games/chess/Move.inl"
