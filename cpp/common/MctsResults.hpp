#pragma once

#include <common/DerivedTypes.hpp>

namespace common {

template<typename GameState>
struct MctsResults_ {
  using GameStateTypes = GameStateTypes_<GameState>;

  using ActionMask = typename GameStateTypes::ActionMask;
  using LocalPolicyCountDistr = typename GameStateTypes::LocalPolicyCountDistr;
  using LocalPolicyProbDistr = typename GameStateTypes::LocalPolicyProbDistr;
  using ValueProbDistr = typename GameStateTypes::ValueProbDistr;

  ActionMask valid_actions;
  LocalPolicyCountDistr counts;
  LocalPolicyProbDistr policy_prior;
  ValueProbDistr win_rates;
  ValueProbDistr value_prior;
};

}  // namespace common
