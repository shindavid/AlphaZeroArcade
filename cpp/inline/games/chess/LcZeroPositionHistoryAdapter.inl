#include <games/chess/LcZeroPositionHistoryAdapter.hpp>

namespace chess {

inline void LcZeroPositionHistoryAdapter::clear() { history_.positions_.clear(); }

template <typename R>
void LcZeroPositionHistoryAdapter::initialize(R) {
  // 0, 1 constants match usage in lc0/src/neural/encoder_test.cc
  history_.Reset(lczero::ChessBoard::kStartposBoard, 0, 1);
}

inline void LcZeroPositionHistoryAdapter::update(const lczero::Position& state) {
  // Adapted from lczero::PositionHistory::Append(). We can't use that function directly
  // because that method accepts a Move (to be applied to the last Position).
  history_.positions_.push_back(state);
  int cycle_length;
  int repetitions = history_.ComputeLastMoveRepetitions(&cycle_length);
  history_.positions_.back().SetRepetitions(repetitions, cycle_length);
}

inline void LcZeroPositionHistoryAdapter::undo() {
  DEBUG_ASSERT(!history_.positions_.empty());
  history_.Pop();
}

inline const lczero::Position& LcZeroPositionHistoryAdapter::current() const {
  DEBUG_ASSERT(!history_.positions_.empty());
  return history_.positions_.back();
}

inline lczero::Position& LcZeroPositionHistoryAdapter::current() {
  DEBUG_ASSERT(!history_.positions_.empty());
  return history_.positions_.back();
}

inline auto LcZeroPositionHistoryAdapter::begin() const {
  auto it1 = history_.positions_.begin();
  auto it2 = history_.positions_.end() - kNumPreviousStatesToEncode;
  return std::max(it1, it2);
}

inline auto LcZeroPositionHistoryAdapter::begin() {
  auto it1 = history_.positions_.begin();
  auto it2 = history_.positions_.end() - kNumPreviousStatesToEncode;
  return std::max(it1, it2);
}

inline auto LcZeroPositionHistoryAdapter::end() const { return history_.positions_.end(); }

inline auto LcZeroPositionHistoryAdapter::end() { return history_.positions_.end(); }

}  // namespace chess
