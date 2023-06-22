#pragma once

#include <iostream>

#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/MctsResults.hpp>

namespace common {

/*
 * This class can be template-specialized to dump the results of MCTS.
 */
template<GameStateConcept GameState>
struct MctsResultsDumper {
  using GameStateTypes = common::GameStateTypes<GameState>;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using MctsResults = common::MctsResults<GameState>;

  static void dump(const LocalPolicyArray& action_policy, const MctsResults& results) {
    printf("TODO: Specialize MctsResultsDumper<%s>\n", util::get_typename<GameState>().c_str());
  }
};

}  // namespace common
