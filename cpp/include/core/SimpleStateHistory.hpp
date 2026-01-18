#pragma once

#include "util/StaticCircularBuffer.hpp"

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
template <typename State, int kNumPastStatesNeeded>
class SimpleStateHistory {
 public:
  constexpr static int kHistoryLength = kNumPastStatesNeeded;
  void clear() { buf_.clear(); }

  /*
   * Clears the history, and then adds the initial state according to the Rules.
   *
   * Rules is passed as an argument here to make the call-site more readable. We could have passed
   * Rules as a template parameter, but then we would need to use the "template" keyword at the
   * call-site, which is distasteful.
   */
  template <typename Rules>
  void initialize(Rules);

  /*
   * Push back a copy of the most recent state, and return a reference to it.
   */
  State& extend();

  /*
   * Push back the given state.
   */
  void update(const State& state);

  /*
   * Undo the most recent update() call. Assumes that the history is not empty, and that any two
   * undo() calls will have an update() call in between them.
   *
   * This is used in a very specialized context within MCTS.
   */
  void undo();

  /*
   * Return a reference to the most recent state in the history. Assumes that the history is not
   * empty.
   */
  const State& current() const;
  State& current();

  auto begin() const;
  auto end() const { return buf_.end(); }
  auto begin();
  auto end() { return buf_.end(); }

 private:
  /*
   * We need kNumPastStatesNeeded + 1 for the past states + the current state.
   *
   * Then, we need another +1 to support the undo() method.
   */
  static constexpr size_t kMaxSize = kNumPastStatesNeeded + 2;
  using buffer_t = util::StaticCircularBuffer<State, kMaxSize>;

  buffer_t buf_;
};

}  // namespace core

#include "inline/core/SimpleStateHistory.inl"
