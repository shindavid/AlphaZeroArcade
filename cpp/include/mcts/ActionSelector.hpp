#pragma once

#include "core/BasicTypes.hpp"
#include "search/ManagerParams.hpp"
#include "search/SearchParams.hpp"

namespace mcts {

template <typename Traits>
struct ActionSelector {
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using ManagerParams = search::ManagerParams<Game>;
  using LocalPolicyArray = Node::LocalPolicyArray;

  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;
  static constexpr float eps = 1e-6;  // needed when N == 0

  ActionSelector(const ManagerParams& manager_params, const search::SearchParams& search_params,
                 const Node* node, bool is_root);

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
