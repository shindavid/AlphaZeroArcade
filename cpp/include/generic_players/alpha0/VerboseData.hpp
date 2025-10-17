#pragma once

#include "core/BasicTypes.hpp"
#include "search/VerboseDataBase.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace generic::alpha0 {

template <search::concepts::Traits Traits_>
struct VerboseData : public VerboseDataBase {
  using Traits = Traits_;
  using Game = Traits::Game;
  using IO = Game::IO;
  using PolicyTensor = Game::Types::PolicyTensor;
  using SearchResults = ::alpha0::SearchResults<Game>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;

  VerboseData(int n_rows_to_display) : n_rows_to_display_(n_rows_to_display) {}

  PolicyTensor action_policy;
  SearchResults mcts_results;

  boost::json::object to_json() const;
  void to_terminal_text() const;
  void set(const PolicyTensor& policy, const SearchResults& results);

 private:
  int n_rows_to_display_ = -1;
  auto build_action_data() const;
};

}  // namespace generic::alpha0

#include "inline/generic_players/alpha0/VerboseData.inl"
