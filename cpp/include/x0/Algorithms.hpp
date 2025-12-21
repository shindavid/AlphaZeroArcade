#pragma once

#include "search/concepts/TraitsConcept.hpp"
#include "x0/SearchResults.hpp"

namespace x0 {

// CRTP base class
//
// This allows us to effectively have x0::Algorithms methods invoke {alpha0,beta0}::Algorithms
// methods.
//
// To make this work, we need to follow the discipline of always invoking Derived::func() rather
// than simply func() within x0::Algorithms methods.
template <search::concepts::Traits Traits, typename Derived>
class AlgorithmsBase {
 public:
  using Game = Traits::Game;
  using SearchResults = x0::SearchResults<Game>;
  using PolicyTensor = Game::Types::PolicyTensor;

  static bool validate_and_symmetrize_policy_target(const SearchResults* mcts_results,
                                                    PolicyTensor& target);
};

template <search::concepts::Traits Traits>
struct Algorithms : public AlgorithmsBase<Traits, Algorithms<Traits>> {};

}  // namespace x0

#include "inline/x0/Algorithms.inl"
