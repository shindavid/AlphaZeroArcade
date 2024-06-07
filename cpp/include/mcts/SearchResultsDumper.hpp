#pragma once

#include <iostream>

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <mcts/SearchResults.hpp>
#include <util/CppUtil.hpp>

namespace mcts {

/*
 * This class can be template-specialized to dump the results of MCTS.
 */
template <core::concepts::Game Game>
struct SearchResultsDumper {
  using GameStateTypes = core::GameStateTypes<GameState>;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using SearchResults = mcts::SearchResults<GameState>;

  static void dump(const LocalPolicyArray& action_policy, const SearchResults& results) {
    printf("TODO: Specialize SearchResultsDumper<%s>\n", util::get_typename<GameState>().c_str());
  }
};

}  // namespace mcts
