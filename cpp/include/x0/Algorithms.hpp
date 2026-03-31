#pragma once

#include "core/ActionSymmetryTable.hpp"
#include "search/GeneralContext.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace x0 {

// Base class of {alpha0,beta0}::Algorithms
template <search::concepts::Traits Traits>
class Algorithms {
 public:
  using Game = Traits::Game;
  using Edge = Traits::Edge;
  using SearchResults = Traits::SearchResults;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using SearchContext = search::SearchContext<Traits>;
  using GeneralContext = search::GeneralContext<Traits>;
  using LookupTable = search::LookupTable<Traits>;

  using State = Game::State;
  using TraitsTypes = search::TraitsTypes<Traits>;
  using Node = TraitsTypes::Node;

  using EvalSpec = Traits::EvalSpec;
  using ActionSymmetryTable = core::ActionSymmetryTable<EvalSpec>;
  using Symmetries = EvalSpec::Symmetries;
  using InputFrame = EvalSpec::InputFrame;

  static void print_visit_info(const SearchContext&);

 protected:
  static bool validate_and_symmetrize_policy_target(const SearchResults* mcts_results,
                                                    PolicyTensor& target);
  static void load_action_symmetries(const GeneralContext&, const Node* root, SearchResults&);
  static ActionValueTensor apply_mask(const ActionValueTensor&, const PolicyTensor& mask,
                                      float invalid_value = -1.0f);
};

}  // namespace x0

#include "inline/x0/Algorithms.inl"
