#pragma once

#include <chess-library/include/chess.hpp>
#include "core/ArrayMoveList.hpp"
#include "core/BasicTypes.hpp"
#include "games/chess/Constants.hpp"

#include <algorithm>
#include <random>

namespace a0achess {

class Move : public chess::Move {
 public:
  Move() = default;
  Move(const chess::Move& move, core::game_phase_t phase) : chess::Move(move), phase_(phase) {}

  auto operator<=>(const Move& m) const { return move() <=> m.move(); }
  core::game_phase_t phase() const { return phase_; }

 private:
  core::game_phase_t phase_;
};

class MoveList : public core::ArrayMovelist<Move, kMaxBranchingFactor> {
 public:
  static constexpr bool kSortedByMove = false;
  using core::ArrayMovelist<Move, kMaxBranchingFactor>::ArrayMovelist;

  int count() const { return size(); }  // TODO: rename count() to size in the MoveList interface
  Move get_random(std::mt19937& prng) const;  // assumes !empty()
  bool contains(const Move& m) const { return std::find(begin(), end(), m) != end(); }

  std::string to_string() const;  // should be called to_json()
};

}  // namespace a0achess

#include "inline/games/chess/Move.inl"
