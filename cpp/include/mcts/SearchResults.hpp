#pragma once

namespace mcts {

template <typename Game>
struct SearchResults {
  using ActionMask = typename Game::ActionMask;
  using PolicyTensor = typename Game::PolicyTensor;
  using ValueArray = typename Game::ValueArray;

  ActionMask valid_actions;
  PolicyTensor counts;
  PolicyTensor policy_target;
  PolicyTensor policy_prior;
  ValueArray win_rates;
  ValueArray value_prior;
  bool provably_lost = false;
};

}  // namespace mcts
