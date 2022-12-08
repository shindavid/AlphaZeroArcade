#pragma once

#include <common/DerivedTypes.hpp>

namespace common {

/*
 * TODO: use local distrs rather than global distrs. This will yield space savings for games where the number
 * of global actions is much greater than the number of local actions, like chess.
 */
template<typename GameState>
struct MctsResults_ {
  using GameStateTypes = GameStateTypes_<GameState>;
  using GlobalPolicyCountDistr = typename GameStateTypes::GlobalPolicyCountDistr;
  using GlobalPolicyProbDistr = typename GameStateTypes::GlobalPolicyProbDistr;
  using ValueProbDistr = typename GameStateTypes::ValueProbDistr;

  GlobalPolicyCountDistr counts;
  GlobalPolicyProbDistr policy_prior;
  ValueProbDistr win_rates;
  ValueProbDistr value_prior;
};

}  // namespace common
