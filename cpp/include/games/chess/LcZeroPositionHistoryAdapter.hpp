#pragma once

#include "lc0/chess/position.h"

namespace chess {

/*
 * A thin-wrapper around lczero::PositionHistory.
 *
 * We require friend-access to lczero::PositionHistory to implement the GameStateHistory concept,
 * which is why we define this outside of the Game class, as c++ does not allow forward-declarations
 * of nested classes.
 */
class LcZeroPositionHistoryAdapter {
 public:
  void clear();
  template <typename R>
  void initialize(R);
  void update(const lczero::Position&);
  void undo();
  const lczero::Position& current() const;
  lczero::Position& current();
  auto begin() const;
  auto begin();
  auto end() const;
  auto end();

  lczero::PositionHistory& lc0_history() { return history_; }
  const lczero::PositionHistory& lc0_history() const { return history_; }

 private:
  lczero::PositionHistory history_;
};

}  // namespace chess

#include "inline/games/chess/LcZeroPositionHistoryAdapter.inl"
