#pragma once

#include "betazero/ManagerParams.hpp"
#include "core/BasicTypes.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
struct ActionSelector {
  using Game = Traits::Game;
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using EvalSpec = Traits::EvalSpec;
  using LookupTable = search::LookupTable<Traits>;
  using ManagerParams = beta0::ManagerParams<EvalSpec>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;

  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;
  static constexpr float eps = 1e-6;  // needed when N == 0

  ActionSelector(const LookupTable& lookup_table, const ManagerParams& manager_params,
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

}  // namespace beta0

#include "inline/betazero/ActionSelector.inl"
