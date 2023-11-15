#pragma once

#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/TreeData.hpp>
#include <mcts/TreeTraversalThread.hpp>
#include <mcts/TypeDefs.hpp>

#include <boost/filesystem.hpp>

namespace mcts {
/*
 * See documentation for TreeTraversalThread.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class SearchThread : public TreeTraversalThread<GameState, Tensorizor> {
 public:
  using base_t = TreeTraversalThread<GameState, Tensorizor>;
  using GameStateTypes = typename base_t::GameStateTypes;
  using NNEvaluation = typename base_t::NNEvaluation;
  using NNEvaluationService = typename base_t::NNEvaluationService;
  using Node = typename base_t::Node;
  using TreeData = typename base_t::TreeData;
  using ValueArray = typename base_t::ValueArray;
  using edge_t = typename base_t::edge_t;

  SearchThread(TreeData*, NNEvaluationService*, const ManagerParams*);

  void set_search_params(const SearchParams* search_params) {
    this->search_params_ = search_params;
  }

 protected:
  void loop();
  void search(Node* root, Node* node, edge_t* edge, move_number_t move_number);
};

}  // namespace mcts

#include <mcts/inl/SearchThread.inl>
