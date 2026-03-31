#pragma once

namespace core {

// The result when querying the rules engine for a given state.
template <typename Types>
class RulesResult {
 public:
  using GameResultTensor = Types::GameResultTensor;
  using MoveList = Types::MoveList;

  static RulesResult make_terminal(const GameResultTensor& outcome);
  static RulesResult make_nonterminal(const MoveList& valid_moves);

  bool is_terminal() const { return terminal_; }
  const GameResultTensor& outcome() const;
  const MoveList& valid_moves() const;

 private:
  GameResultTensor outcome_;  // Only valid if terminal
  MoveList valid_moves_;
  bool terminal_;  // Must be equal to !valid_moves.empty()
};

}  // namespace core

#include "inline/core/RulesResult.inl"
