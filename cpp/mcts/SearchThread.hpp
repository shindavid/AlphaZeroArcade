#pragma once

#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/SharedData.hpp>
#include <mcts/TreeTraversalThread.hpp>

#include <boost/filesystem.hpp>

namespace mcts {
/*
 * See documentation for TreeTraversalThread.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class SearchThread : public TreeTraversalThread<GameState, Tensorizor> {
 public:
  using base_t = TreeTraversalThread<GameState, Tensorizor>;
  using NNEvaluationService = mcts::NNEvaluationService<GameState, Tensorizor>;
  using SharedData = mcts::SharedData<GameState, Tensorizor>;

  SearchThread(SharedData*, NNEvaluationService*, const ManagerParams*);

  void set_search_params(const SearchParams* search_params) {
    search_params_ = search_params;
  }

 protected:
  void loop();
  void visit(Node* tree, edge_t* edge, move_number_t move_number);
};

}  // namespace mcts

#include <mcts/inl/SearchThread.inl>
