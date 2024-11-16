#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchParams.hpp>
#include <util/EigenUtil.hpp>

namespace mcts {

template <core::concepts::Game Game>
struct ActionSelector {
  using ManagerParams = mcts::ManagerParams<Game>;
  using Node = mcts::Node<Game>;
  using LocalPolicyArray = Node::LocalPolicyArray;

  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;
  static constexpr float eps = 1e-6;  // needed when N == 0

  void load(const ManagerParams& manager_params, const SearchParams& search_params,
            const Node* node, bool is_root);

  LocalPolicyArray P;
  LocalPolicyArray Q;    // (virtualized) value
  LocalPolicyArray QLB;  // Q lower bound
  LocalPolicyArray QUB;  // Q upper bound
  LocalPolicyArray PW;   // provably-winning
  LocalPolicyArray PL;   // provably-losing
  LocalPolicyArray E;    // edge count
  LocalPolicyArray mE;   // masked edge count
  LocalPolicyArray RN;   // real node count
  LocalPolicyArray VN;   // virtual node count
  LocalPolicyArray FPU;  // FPU
  LocalPolicyArray PUCT;

  core::seat_index_t cp;
  bool loaded = false;
};

}  // namespace mcts

#include <inline/mcts/ActionSelector.inl>
