#pragma once

namespace core {

// The result when querying the rules engine for a given state.
template <typename Types>
class RulesResult {
 public:
  using GameOutcome = Types::GameOutcome;
  using MoveSet = Types::MoveSet;

  RulesResult() = default;
  RulesResult(const GameOutcome& outcome);
  RulesResult(const MoveSet& valid_moves);

  bool is_terminal() const { return terminal_; }
  const GameOutcome& outcome() const;
  const MoveSet& valid_moves() const;

 private:
  GameOutcome outcome_;  // Only valid if terminal
  MoveSet valid_moves_;
  bool terminal_;  // Must be equal to !valid_moves.empty()
};

}  // namespace core

#include "inline/core/RulesResult.inl"
