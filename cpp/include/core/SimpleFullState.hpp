#pragma once

namespace core {

/*
 * Each Game must define a BaseState and a FullState inner-class. The BaseState should be a
 * simple POD (piece-of-data) struct. The FullState potentially contains additional information that
 * is needed to apply the game rules, but which perhaps cannot fit in a POD struct.
 *
 * This class can be used for games where the FullState does not need anything extra beyond the
 * BaseState.
 *
 * Games where this may not be enough:
 *
 * - Chess: Prior positions are needed to determine if a draw by repetition has occurred.
 * - Go: Prior positions are needed for superko rules.
 *
 * For chess/go, a specialized FullState class with a Zobrist hash table would be more appropriate.
 */
template<typename BaseState>
class SimpleFullState {
 public:
  using EvalKey = BaseState;
  using MCTSKey = BaseState;

  const EvalKey& eval_key() const { return base_; }
  const MCTSKey& mcts_key() const { return base_; }
  const BaseState& base() const { return base_; }
  BaseState& base() { return base_; }
  void update(const BaseState& base) { base_ = base; }

 private:
  BaseState base_;
};

}  // namespace core
