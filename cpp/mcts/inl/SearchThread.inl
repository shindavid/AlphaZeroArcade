#include <mcts/SearchThread.hpp>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
SearchThread<GameState, Tensorizor>::SearchThread(SharedData* shared_data,
                                                  NNEvaluationService* nn_eval_service,
                                                  const ManagerParams* manager_params)
    : base_t(kSearchMode, manager_params->profiling_dir(), 0) {
  this->shared_data_ = shared_data;
  this->nn_eval_service_ = nn_eval_service;
  this->manager_params_ = manager_params;
  this->thread_ = new std::thread([this] { loop(); });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::loop() {
  while (true) {
    this->shared_data_->wait_for_search_activation();
    if (this->shared_data_->shutdown_initiated()) return;

    this->search_path_.clear();
    Node* root = this->shared_data_->root_node.get();
    search(root, root, nullptr, this->shared_data_->move_number);

    if (root->stats(kSearchMode).total_count() > this->search_params_->tree_size_limit) {
      this->shared_data_->deactivate_search();
    }

    this->dump_profiling_stats();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::search(Node* root, Node* tree, edge_t* edge,
                                                 move_number_t move_number) {
  this->search_path_.emplace_back(tree, edge);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(this->thread_id_);
    if (edge) {
      printer << __func__ << util::std_array_to_string(edge->action(), "(", ",", ")");
    } else {
      printer << __func__ << "()";
    }
    printer << " " << this->search_path_str() << " cp=" << (int)tree->stable_data().current_player
            << std::endl;
  }

  const auto& stable_data = tree->stable_data();
  const auto& outcome = stable_data.outcome;
  if (GameStateTypes::is_terminal_outcome(outcome)) {
    this->backprop(outcome, kTerminal);
    return;
  }

  if (!this->shared_data_->search_active()) return;  // short-circuit

  constexpr int kPrefetchFailLimit = 400;
  bool eval_available = this->shared_data_->wait_for_eval(root, tree, kPrefetchFailLimit);
  if (!eval_available) {
    this->shared_data_->reset_prefetch_threads();
    eval_available = this->shared_data_->wait_for_eval(root, tree, kPrefetchFailLimit);
    util::release_assert(eval_available);
  }

  bool first_visit = tree->stats(kSearchMode).real_count == 0;
  NNEvaluation* evaluation = tree->evaluation_data().ptr.load().get();

  if (first_visit) {
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer(this->thread_id_);
      printer << "hit leaf node" << std::endl;
    }
    this->backprop(evaluation->value_prob_distr(), kNonterminal);
  } else {
    core::action_index_t action_index = this->get_best_action_index(tree, evaluation);
    edge_t* edge = this->shared_data_->wait_for_edge(root, tree, action_index, kPrefetchFailLimit);
    if (!edge) {
      this->shared_data_->reset_prefetch_threads();
      edge = this->shared_data_->wait_for_edge(root, tree, action_index, kPrefetchFailLimit);
      util::release_assert(edge);
    }

    int edge_count = edge->count(kSearchMode);
    int child_count = edge->child()->stats(kSearchMode).real_count;
    if (edge_count < child_count) {
      this->short_circuit_backprop(edge);
    } else {
      search(root, edge->child().get(), edge, move_number + 1);
    }
  }
}

}  // namespace mcts
