#pragma once

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>

namespace mcts {

template <core::GameStateConcept GameState>
struct SearchResults {
  using GameStateTypes = core::GameStateTypes<GameState>;

  using ActionMask = typename GameStateTypes::ActionMask;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueArray = typename GameStateTypes::ValueArray;

  ActionMask valid_actions;
  PolicyTensor counts;
  PolicyTensor policy_target;
  LocalPolicyArray policy_prior;
  ValueArray win_rates;
  ValueArray value_prior;
  bool provably_lost = false;
};

}  // namespace mcts
