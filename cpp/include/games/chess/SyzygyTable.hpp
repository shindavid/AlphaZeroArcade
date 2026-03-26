#pragma once

#include "core/BasicTypes.hpp"
#include "games/chess/GameState.hpp"

#include <cstdint>

namespace a0achess {

class SyzygyTable {
 public:
  enum Result : int8_t { kUnknownResult, kDraw, kBlackWins, kWhiteWins };

  static constexpr int kMaxNumPieces = 5;

  // raises a util::CleanException if the syzygy tables are not available. The exception message
  // will include instructions to run setup_wizard.py from outside the Docker container in order to
  // download the tables.
  static SyzygyTable& instance();

  // If the position is not found (i.e., if the state has > kMaxNumPieces or if the position is
  // illegal), then returns kUnknownResult. Otherwise, returns the game-theoretic result of the
  // position. If action is not nullptr, then it is set to the best action in the position.
  //
  // If the position is a win for the side to move, the best action is the one that leads to the
  // fastest win. In the case of a losing position, the best action is the one that leads to the
  // slowest loss. In the event that multiple actions qualify as the best action, one of them is
  // chosen arbitrarily.
  Result lookup(const GameState& state, core::action_t* action = nullptr) const;
};

}  // namespace a0achess

#include "inline/games/chess/SyzygyTable.inl"
