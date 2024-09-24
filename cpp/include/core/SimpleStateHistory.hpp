#pragma once

#include <util/Exception.hpp>

#include <boost/circular_buffer.hpp>

namespace core {

/*
 * SimpleStateHistory can be used in a Game where the game rules only care about the current state,
 * and not the history of states.
 *
 * The kNumPastStatesNeeded parameter might be nonzero if the neural network encodes past states as
 * part of the input tensor. If that is not the case, then there is no reason for the value to be
 * nonzero.
 *
 * Chess is an example of a game where SimpleStateHistory would *not* be appropriate, because of the
 * threefold repetition rule and the fifty-move rule.
 *
 * Under the hood, we actually store kNumPastStatesNeeded + 2 states. One +1 factor is for the
 * current state. The other +1 factor is to support an undo() method.
 */
template <typename BaseState, int kNumPastStatesNeeded>
class SimpleStateHistory {
 public:
  SimpleStateHistory() : buf_(kMaxSize) {}

  void clear() { buf_.clear(); }
  void update(const BaseState& state) { buf_.push_back(state); }
  void undo() { buf_.pop_back(); }
  const BaseState& current() const { return buf_.back(); }
  auto begin() const {
    auto it = buf_.begin();
    if (buf_.size() == kMaxSize) {
      ++it;  // skip the oldest state, it's only there to support undo()
    }
    return it;
  }
  auto end() const { return buf_.end(); }
  auto begin() {
    auto it = buf_.begin();
    if (buf_.size() == kMaxSize) {
      ++it;  // skip the oldest state, it's only there to support undo()
    }
    return it;
  }
  auto end() { return buf_.end(); }

 private:
  /*
   * We need kNumPastStatesNeeded + 1 for the past states + the current state.
   *
   * Then, we need another +1 to support the temp_push() method.
   */
  static constexpr size_t kMaxSize = kNumPastStatesNeeded + 2;

  /*
   * TODO: we use an array for convenience. Consider using a circular buffer instead to avoid
   * copying within append(). If switching to circular buffer, we just need to be careful about
   * iterator mechanics.
   */
  using buffer_t = boost::circular_buffer<BaseState>;

  buffer_t buf_;
};

}  // namespace core
