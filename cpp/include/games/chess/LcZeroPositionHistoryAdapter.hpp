#pragma once

#include "lc0/chess/position.h"

namespace chess {

/*
 * A thin-wrapper around lczero::PositionHistory.
 *
 * TODO: we do not plan to use this class anymore, and will instead rely on specially customized
 * machinery that meets the requirements of Game::Symmetries.
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
