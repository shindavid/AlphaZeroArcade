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

  VerboseData(const PolicyTensor& p, const SearchResults& s, int n)
      :  action_policy(p), mcts_results(s), n_rows_to_display_(n) {}

  PolicyTensor action_policy;
  SearchResults mcts_results;

  boost::json::object to_json() const;
  void to_terminal_text() const;

 private:
  const int n_rows_to_display_;
  auto build_action_data() const;
};

}  // namespace alpha0

#include "inline/alpha0/VerboseData.inl"
