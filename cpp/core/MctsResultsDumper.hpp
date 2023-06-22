#pragma once

#include <iostream>

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/MctsResults.hpp>

namespace core {

/*
 * This class can be template-specialized to dump the results of MCTS.
 */
template<GameStateConcept GameState>
struct MctsResultsDumper {
  using GameStateTypes = core::GameStateTypes<GameState>;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using MctsResults = core::MctsResults<GameState>;

  static void dump(const LocalPolicyArray& action_policy, const MctsResults& results) {
    printf("TODO: Specialize MctsResultsDumper<%s>\n", util::get_typename<GameState>().c_str());
  }
};

}  // namespace core
