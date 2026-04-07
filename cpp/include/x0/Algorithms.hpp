#pragma once

#include "core/ActionSymmetryTable.hpp"
#include "search/GeneralContext.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

namespace x0 {

// Base class of {alpha0,beta0}::Algorithms
template <search::concepts::SearchSpec SearchSpec>
class Algorithms {
 public:
  using Game = SearchSpec::Game;
  using Edge = SearchSpec::Edge;
  using SearchResults = SearchSpec::SearchResults;
  using SearchContext = search::SearchContext<SearchSpec>;
  using GeneralContext = search::GeneralContext<SearchSpec>;
  using LookupTable = search::LookupTable<SearchSpec>;

  using State = Game::State;
  using Node = SearchSpec::Node;

  using EvalSpec = SearchSpec::EvalSpec;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using ActionValueEncoding = TensorEncodings::ActionValueEncoding;
  using PolicyTensor = PolicyEncoding::Tensor;
  using ActionValueTensor = ActionValueEncoding::Tensor;
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
