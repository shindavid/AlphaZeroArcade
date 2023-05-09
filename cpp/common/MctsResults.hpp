#pragma once

#include <common/DerivedTypes.hpp>

namespace common {

template<typename GameState>
struct MctsResults {
  using GameStateTypes = common::GameStateTypes<GameState>;

  using ActionMask = typename GameStateTypes::ActionMask;
  using PolicyEigenTensor = typename GameStateTypes::PolicyTensor::EigenType;
  using LocalPolicyProbDistr = typename GameStateTypes::LocalPolicyProbDistr;
  using ValueProbDistr = typename GameStateTypes::ValueProbDistr;

  ActionMask valid_actions;
  PolicyEigenTensor counts;
  LocalPolicyProbDistr policy_prior;
  ValueProbDistr win_rates;
  ValueProbDistr value_prior;
};

}  // namespace common
