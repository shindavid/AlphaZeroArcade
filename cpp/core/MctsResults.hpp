#pragma once

#include <core/DerivedTypes.hpp>

namespace core {

template<typename GameState>
struct MctsResults {
  using GameStateTypes = core::GameStateTypes<GameState>;

  using ActionMask = typename GameStateTypes::ActionMask;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using ValueArray = typename GameStateTypes::ValueArray;

  ActionMask valid_actions;
  PolicyTensor counts;
  LocalPolicyArray policy_prior;
  ValueArray win_rates;
  ValueArray value_prior;
};

}  // namespace core
