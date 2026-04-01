#pragma once

#include "core/BasicTypes.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
struct PuctCalculator {
  using Game = Traits::Game;
  using Edge = Traits::Edge;
  using EvalSpec = Traits::EvalSpec;
  using LookupTable = search::LookupTable<Traits>;
  using ManagerParams = Traits::ManagerParams;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;

  using TraitsTypes = search::TraitsTypes<Traits>;
  using Node = TraitsTypes::Node;

  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;
  static constexpr float eps = 1e-6;  // needed when N == 0

  PuctCalculator(const LookupTable& lookup_table, const ManagerParams& manager_params,
                 const search::SearchParams& search_params, const Node* node, bool is_root);

  core::seat_index_t seat;
  LocalPolicyArray P;
  LocalPolicyArray Q;    // (virtualized) value
  LocalPolicyArray W;    // uncertainty
  LocalPolicyArray E;    // edge count
  LocalPolicyArray mE;   // masked edge count
  LocalPolicyArray N;    // real node count
  LocalPolicyArray PUCT;
};

}  // namespace beta0

#include "inline/beta0/PuctCalculator.inl"
