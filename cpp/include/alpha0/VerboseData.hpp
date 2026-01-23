#pragma once

#include "alpha0/SearchResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/VerboseDataBase.hpp"

namespace alpha0 {

template <core::concepts::Game Game>
struct VerboseData : public generic::VerboseDataBase {
  using IO = Game::IO;
  using PolicyTensor = Game::Types::PolicyTensor;
  using SearchResults = alpha0::SearchResults<Game>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;

  VerboseData(int n_rows_to_display) : n_rows_to_display_(n_rows_to_display) {}

  PolicyTensor action_policy;
  SearchResults mcts_results;

  boost::json::object to_json() const;
  void to_terminal_text() const;
  void set(const PolicyTensor& policy, const SearchResults& results);

 private:
  const int n_rows_to_display_;
  auto build_action_data() const;
};

}  // namespace alpha0

#include "inline/alpha0/VerboseData.inl"
