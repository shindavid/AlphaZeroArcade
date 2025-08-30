#pragma once

#include "core/BasicTypes.hpp"
#include "mcts/ManagerParams.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"
#include "search/concepts/GraphTraitsConcept.hpp"

namespace mcts {

template <search::concepts::GraphTraits GraphTraits>
struct ActionSelector {
  using Game = GraphTraits::Game;
  using Node = GraphTraits::Node;
  using Edge = GraphTraits::Edge;
  using LookupTable = search::LookupTable<GraphTraits>;
  using ManagerParams = mcts::ManagerParams<Game>;
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

}  // namespace mcts

#include "inline/mcts/ActionSelector.inl"
