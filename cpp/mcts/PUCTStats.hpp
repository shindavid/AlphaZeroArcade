#pragma once

#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchParams.hpp>
#include <util/TorchUtil.hpp>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
struct PUCTStats {
  using Node = mcts::Node<GameState, Tensorizor>;
  using GameStateTypes = core::GameStateTypes<GameState>;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using dtype = torch_util::dtype;

  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;
  static constexpr float eps = 1e-6;  // needed when N == 0

  PUCTStats(const ManagerParams& manager_params, const SearchParams& search_params, const Node* tree, bool is_root);

  core::seat_index_t cp;
  const LocalPolicyArray& P;
  LocalPolicyArray V;  // (virtualized) value
  LocalPolicyArray PW;  // provably-winning
  LocalPolicyArray PL;  // provably-losing
  LocalPolicyArray E;  // edge count
  LocalPolicyArray N;  // real count
  LocalPolicyArray VN;  // virtual count
  LocalPolicyArray PUCT;
};

}  // namespace mcts

#include <mcts/inl/PUCTStats.inl>
