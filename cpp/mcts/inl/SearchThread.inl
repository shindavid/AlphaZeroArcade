#include <mcts/SearchThread.hpp>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
SearchThread<GameState, Tensorizor>::SearchThread(SharedData* shared_data,
                                                  NNEvaluationService* nn_eval_service,
                                                  const ManagerParams* manager_params)
    : base_t(kSearchMode, params.profiling_dir(), 0),
      shared_data_(shared_data),
      nn_eval_service_(nn_eval_service),
      manager_params_(manager_params) {
  thread_ = new std::thread([this] { loop(); });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::loop() {
  while (true) {
    shared_data_->wait_for_search_activation();

    search_path_.clear();
    Node* root = shared_data_->root_node.get();
    visit(root, nullptr, shared_data_->move_number);

    if (root->stats().total_count() > search_params_->tree_size_limit) {
      manager_->remove_work(shared_data_);
    }

    dump_profiling_stats();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::visit(Node* tree, edge_t* edge,
                                                move_number_t move_number) {
  search_path_.emplace_back(tree, edge);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id());
    if (edge) {
      printer << __func__ << util::std_array_to_string(edge->action(), "(", ",", ")");
    } else {
      printer << __func__ << "()";
    }
    printer << " " << search_path_str() << " cp=" << (int)tree->stable_data().current_player
            << std::endl;
  }

  const auto& stable_data = tree->stable_data();
  const auto& outcome = stable_data.outcome;
  if (GameStateTypes::is_terminal_outcome(outcome)) {
    pure_backprop(outcome);
    return;
  }

  if (!shared_data_->search_active()) return;  // short-circuit

  evaluation_result_t data = evaluate(tree);
  NNEvaluation* evaluation = data.evaluation.get();

  if (data.backpropagated_virtual_loss) {
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer(thread_id());
      printer << "hit leaf node" << std::endl;
    }
    backprop_with_virtual_undo(evaluation->value_prob_distr());
  } else {
    auto& children_data = tree->children_data();
    core::action_index_t action_index = get_best_action_index(tree, evaluation);

    edge_t* edge = children_data.find(action_index);
    if (!edge) {
      Action action =
          GameStateTypes::get_nth_valid_action(stable_data.valid_action_mask, action_index);
      auto child = shared_data_->node_cache.fetch_or_create(move_number, tree, action);

      std::unique_lock lock(tree->children_mutex());
      edge = children_data.insert(action, action_index, child);
    }

    int edge_count = edge->count();
    int child_count = edge->child()->stats(traversal_mode_).real_count;
    if (edge_count < child_count) {
      short_circuit_backprop(edge);
    } else {
      visit(edge->child().get(), edge, move_number + 1);
    }
  }
}

}  // namespace mcts
