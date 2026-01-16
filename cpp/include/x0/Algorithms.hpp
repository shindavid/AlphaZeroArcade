#pragma once

#include "search/GeneralContext.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "x0/SearchResults.hpp"

namespace x0 {

// Base class of {alpha0,beta0}::Algorithms
template <search::concepts::Traits Traits>
class Algorithms {
 public:
  using Game = Traits::Game;
  using Edge = Traits::Edge;
  using SearchResults = x0::SearchResults<Game>;
  using PolicyTensor = Game::Types::PolicyTensor;
  using SearchContext = search::SearchContext<Traits>;
  using GeneralContext = search::GeneralContext<Traits>;
  using LookupTable = search::LookupTable<Traits>;

  using State = Game::State;
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;
  using TraitsTypes = search::TraitsTypes<Traits>;
  using Node = TraitsTypes::Node;

  static void print_visit_info(const SearchContext&);

 protected:
  static bool validate_and_symmetrize_policy_target(const SearchResults* mcts_results,
                                                    PolicyTensor& target);
  static void load_action_symmetries(const GeneralContext&, const Node* root,
                                     core::action_t* actions, SearchResults&);
};

}  // namespace x0

#include "inline/x0/Algorithms.inl"
