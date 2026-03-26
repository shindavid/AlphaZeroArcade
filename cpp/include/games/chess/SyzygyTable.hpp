#pragma once

#include "core/BasicTypes.hpp"
#include "games/chess/GameState.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

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

  // Fast, thread-safe WDL probe. Returns the game-theoretic result of the position, ignoring the
  // half-move clock (i.e., the 50-move rule is not considered). Returns kUnknownResult if the
  // position has > kMaxNumPieces or has castling rights.
  //
  // This is appropriate for use during tree search. In a well-designed engine, the table will be
  // consulted as soon as the piece count drops to <= kMaxNumPieces, at which point the half-move
  // clock will be 0, making the 50-move rule irrelevant.
  Result fast_lookup(const GameState& state) const;

  // Slow root probe. Returns the game-theoretic result of the position, correctly accounting for
  // the half-move clock (50-move rule). Also returns the best action via the output parameter.
  //
  // The best action is the one leading to the fastest win (or slowest loss). If multiple actions
  // qualify, one is chosen arbitrarily.
  //
  // WARNING: This is significantly slower than fast_lookup(), and acquires a mutex (the underlying
  // Fathom DTZ probe is not thread-safe). Do not use this in a tree-search context.
  Result slow_lookup(const GameState& state, core::action_t* action) const;

 private:
  SyzygyTable();
  mutable mit::mutex mutex_;
};

}  // namespace a0achess

#include "inline/games/chess/SyzygyTable.inl"
