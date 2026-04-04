#pragma once

namespace core {

// The result when querying the rules engine for a given state.
template <typename Types>
class RulesResult {
 public:
  using GameResultTensor = Types::GameResultTensor;
  using MoveSet = Types::MoveSet;

  static RulesResult make_terminal(const GameResultTensor& outcome);
  static RulesResult make_nonterminal(const MoveSet& valid_moves);

  bool is_terminal() const { return terminal_; }
  const GameResultTensor& outcome() const;
  const MoveSet& valid_moves() const;

 private:
  GameResultTensor outcome_;  // Only valid if terminal
  MoveSet valid_moves_;
  bool terminal_;  // Must be equal to !valid_moves.empty()
};

}  // namespace core

#include "inline/core/RulesResult.inl"
