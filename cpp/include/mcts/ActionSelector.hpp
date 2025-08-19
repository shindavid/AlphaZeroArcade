#pragma once

#include "core/BasicTypes.hpp"
#include "mcts/ManagerParams.hpp"
#include "mcts/Node.hpp"
#include "mcts/SearchParams.hpp"

namespace mcts {

template <typename Traits>
struct ActionSelector {
  using Game = Traits::Game;
  using ManagerParams = mcts::ManagerParams<Traits>;
  using Node = mcts::Node<Traits>;
  using LocalPolicyArray = Node::LocalPolicyArray;

  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;
  static constexpr float eps = 1e-6;  // needed when N == 0

  ActionSelector(const ManagerParams& manager_params, const SearchParams& search_params,
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
