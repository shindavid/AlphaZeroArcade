#include "core/SimpleStateHistory.hpp"

#include "util/Exceptions.hpp"

namespace core {

template <typename State, int kNumPastStatesNeeded>
template <typename Rules>
void SimpleStateHistory<State, kNumPastStatesNeeded>::initialize(Rules) {
  clear();
  State state;
  Rules::init_state(state);
  buf_.push_back(state);
}

template <typename State, int kNumPastStatesNeeded>
State& SimpleStateHistory<State, kNumPastStatesNeeded>::extend() {
  DEBUG_ASSERT(!buf_.empty());
  update(buf_.back());
  return buf_.back();
}

template <typename State, int kNumPastStatesNeeded>
void SimpleStateHistory<State, kNumPastStatesNeeded>::update(const State& state) {
  buf_.push_back(state);
}

template <typename State, int kNumPastStatesNeeded>
void SimpleStateHistory<State, kNumPastStatesNeeded>::undo() {
  DEBUG_ASSERT(!buf_.empty());
  buf_.pop_back();
}

template <typename State, int kNumPastStatesNeeded>
const State& SimpleStateHistory<State, kNumPastStatesNeeded>::current() const {
  DEBUG_ASSERT(!buf_.empty());
  return buf_.back();
}

template <typename State, int kNumPastStatesNeeded>
State& SimpleStateHistory<State, kNumPastStatesNeeded>::current() {
  DEBUG_ASSERT(!buf_.empty());
  return buf_.back();
}

template <typename State, int kNumPastStatesNeeded>
auto SimpleStateHistory<State, kNumPastStatesNeeded>::begin() const {
  auto it = buf_.begin();
  if (buf_.size() == kMaxSize) {
    ++it;  // skip the oldest state, it's only there to support undo()
  }
  return it;
}

template <typename State, int kNumPastStatesNeeded>
auto SimpleStateHistory<State, kNumPastStatesNeeded>::begin() {
  auto it = buf_.begin();
  if (buf_.size() == kMaxSize) {
    ++it;  // skip the oldest state, it's only there to support undo()
  }
  return it;
}

template <typename State, int kNumPastStatesNeeded>
SimpleStateHistory<State, kNumPastStatesNeeded>::SimpleStateHistory(
  const ReverseHistory& reverse_history) {
  int k = std::min(kNumPastStatesNeeded + 1, static_cast<int>(reverse_history.size()));
  for (int i = k - 1; i >= 0; --i) {
    buf_.push_back(*reverse_history[i]);
  }
}

}  // namespace core
