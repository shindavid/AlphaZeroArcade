#pragma once

#include "core/BasicTypes.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

namespace alpha0 {

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
  LocalPolicyArray Q;    // (virtualized) value
  LocalPolicyArray PW;   // provably-winning
  LocalPolicyArray PL;   // provably-losing
  LocalPolicyArray E;    // edge count
  LocalPolicyArray mE;   // masked edge count
  LocalPolicyArray RN;   // real node count
  LocalPolicyArray VN;   // virtual node count
  LocalPolicyArray FPU;  // FPU
  LocalPolicyArray PUCT;
};

}  // namespace alpha0

#include "inline/alpha0/PuctCalculator.inl"
