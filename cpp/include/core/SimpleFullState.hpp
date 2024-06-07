#pragma once

namespace core {

/*
 * A simple state that only contains a single position.
 *
 * This is appropriate for games where the current position is all that is needed for
 * rules-calculations and for tensorization.
 *
 * Games where this may not be enough:
 *
 * - Chess: Prior positions are needed to determine if a draw by repetition has occurred.
 * - Go (AlphaGo-style): Prior positions are included in the input tensor.
 */
template<typename StateSnapshot>
class SimpleFullState {
 public:
  using EvalKey = StateSnapshot;
  using MCTSKey = StateSnapshot;

  SimpleFullState(StateSnapshot* start, StateSnapshot* current) : current_(*current) {}
  State() = default;
  const EvalKey& eval_key() const { return current_; }
  const MCTSKey& mcts_key() const { return current_; }
  const StateSnapshot& current() const { return current_; }
  StateSnapshot& current() { return current_; }
  void update(const StateSnapshot& current) { current_ = current; }

 private:
  StateSnapshot current_;
};

}  // namespace core
