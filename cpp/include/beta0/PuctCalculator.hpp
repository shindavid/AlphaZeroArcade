#pragma once

#include "core/BasicTypes.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

namespace beta0 {

template <search::concepts::SearchSpec SearchSpec>
struct PuctCalculator {
  using Game = SearchSpec::Game;
  using Edge = SearchSpec::Edge;
  using EvalSpec = SearchSpec::EvalSpec;
  using LookupTable = search::LookupTable<SearchSpec>;
  using ManagerParams = SearchSpec::ManagerParams;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;

  using Node = SearchSpec::Node;

  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;
  static constexpr float eps = 1e-6;  // needed when N == 0

  PuctCalculator(const LookupTable& lookup_table, const ManagerParams& manager_params,
                 const search::SearchParams& search_params, const Node* node, bool is_root);

  core::seat_index_t seat;
  LocalPolicyArray P;
  LocalPolicyArray Q;   // (virtualized) value
  LocalPolicyArray W;   // uncertainty
  LocalPolicyArray E;   // edge count
  LocalPolicyArray mE;  // masked edge count
  LocalPolicyArray N;   // real node count
  LocalPolicyArray PUCT;
};

}  // namespace beta0

#include "inline/beta0/PuctCalculator.inl"
