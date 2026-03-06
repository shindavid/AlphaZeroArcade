#pragma once

namespace core {

// The result when querying the rules engine for a given state.
template <typename Types>
class RulesResult {
 public:
  using GameResultTensor = Types::GameResultTensor;
  using ActionMask = Types::ActionMask;

  static RulesResult make_terminal(const GameResultTensor& outcome);
  static RulesResult make_nonterminal(const ActionMask& valid_actions);

  bool is_terminal() const { return terminal_; }
  const GameResultTensor& outcome() const { return outcome_; }        // assumes is_terminal()
  const ActionMask& valid_actions() const { return valid_actions_; }  // assumes !is_terminal()

 private:
  GameResultTensor outcome_;  // Only valid if terminal
  ActionMask valid_actions_;
  bool terminal_;  // Must be equal to !valid_actions.empty()
};

}  // namespace core

#include "inline/core/RulesResult.inl"
