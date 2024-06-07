#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchParams.hpp>
#include <util/TorchUtil.hpp>

namespace mcts {

template <core::concepts::Game Game>
struct PUCTStats {
  using Node = mcts::Node<Game>;
  using LocalPolicyArray = typename Game::LocalPolicyArray;

  static constexpr int kMaxBranchingFactor = Game::kMaxBranchingFactor;
  static constexpr float eps = 1e-6;  // needed when N == 0

  PUCTStats(const ManagerParams& manager_params, const SearchParams& search_params,
            const Node* tree, bool is_root);

  core::seat_index_t cp;
  const LocalPolicyArray& P;
  LocalPolicyArray V;   // (virtualized) value
  LocalPolicyArray PW;  // provably-winning
  LocalPolicyArray PL;  // provably-losing
  LocalPolicyArray E;   // edge count
  LocalPolicyArray N;   // real count
  LocalPolicyArray VN;  // virtual count
  LocalPolicyArray PUCT;
};

}  // namespace mcts

#include <inline/mcts/PUCTStats.inl>
