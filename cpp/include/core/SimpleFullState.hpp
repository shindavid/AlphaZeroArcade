#pragma once

namespace core {

/*
 * A simple state that only contains a single snapshot.
 *
 * This is appropriate for games where the current snapshot is all that is needed for
 * rules-calculations.
 *
 * Games where this may not be enough:
 *
 * - Chess: Prior positions are needed to determine if a draw by repetition has occurred.
 * - Go: Prior positions are needed for superko rules.
 *
 * For chess/go, a specialized FullState class with a Zobrist hash table would be more appropriate.
 */
template<typename StateSnapshot>
class SimpleFullState {
 public:
  using EvalKey = StateSnapshot;
  using MCTSKey = StateSnapshot;

  const EvalKey& eval_key() const { return current_; }
  const MCTSKey& mcts_key() const { return current_; }
  const StateSnapshot& current() const { return current_; }
  StateSnapshot& current() { return current_; }
  void update(const StateSnapshot& current) { current_ = current; }

 private:
  StateSnapshot current_;
};

}  // namespace core
