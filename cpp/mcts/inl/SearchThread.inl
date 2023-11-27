#include <mcts/SearchThread.hpp>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
SearchThread<GameState, Tensorizor>::SearchThread(TreeData* tree_data,
                                                  NNEvaluationService* nn_eval_service,
                                                  PrefetchThreadManager* prefetch_manager,
                                                  const ManagerParams* manager_params)
    : base_t(kSearchMode, manager_params->profiling_dir(), 0) {
  this->prefetch_manager_ = prefetch_manager;
  this->tree_data_ = tree_data;
  this->nn_eval_service_ = nn_eval_service;
  this->manager_params_ = manager_params;
  this->thread_ = new std::thread([this] { loop(); });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::loop() {
  while (true) {
    this->profiler_.record(TreeTraversalThreadRegion::kWaitForSearchActivation);
    this->tree_data_->wait_for_search_activation();
    if (this->tree_data_->shutdown_initiated()) return;

    this->search_path_.clear();
    Node* root = this->tree_data_->root_node().get();
    search(root, root, nullptr, this->tree_data_->move_number());

    if (root->stats(kSearchMode).total_count() > this->search_params_->tree_size_limit) {
      this->tree_data_->deactivate_search();
    }

    this->dump_profiling_stats();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::search(Node* root, Node* node, edge_t* edge,
                                                 move_number_t move_number) {
  this->profiler_.record(TreeTraversalThreadRegion::kSearch);
  this->search_path_.emplace_back(node, edge);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(this->thread_id_);
    if (edge) {
      printer << __func__ << util::std_array_to_string(edge->action(), "(", ",", ")");
    } else {
      printer << __func__ << "()";
    }
    printer << " " << this->search_path_str() << " cp=" << (int)node->stable_data().current_player
            << std::endl;
  }

  const auto& stable_data = node->stable_data();
  const auto& outcome = stable_data.outcome;
  if (GameStateTypes::is_terminal_outcome(outcome)) {
    this->backprop(outcome, kTerminal);
    return;
  }

  if (!this->tree_data_->search_active()) return;  // short-circuit

  this->profiler_.record(TreeTraversalThreadRegion::kWaitForEval);

  constexpr int kOverPrefetchLimit = 100;
  bool eval_available = this->tree_data_->wait_for_eval(root, node, kOverPrefetchLimit);
  if (!eval_available) {
    reset();
    eval_available = this->tree_data_->wait_for_eval(root, node, kOverPrefetchLimit);
    util::release_assert(eval_available);
  }

  bool first_visit = node->stats(kSearchMode).real_count == 0;
  NNEvaluation* evaluation = node->evaluation_data().ptr.load().get();

  if (first_visit) {
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer(this->thread_id_);
      printer << "hit leaf node" << std::endl;
    }
    this->backprop(evaluation->value_prob_distr(), kNonterminal);
  } else {
    core::action_index_t action_index = this->get_best_action_index(node, evaluation);
    this->profiler_.record(TreeTraversalThreadRegion::kWaitForEdge);
    edge_t* edge = this->tree_data_->wait_for_edge(root, node, action_index, kOverPrefetchLimit);
    if (!edge) {
      reset();
      edge = this->tree_data_->wait_for_edge(root, node, action_index, kOverPrefetchLimit);
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

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::reset() {
  this->profiler_.record(TreeTraversalThreadRegion::kReset);
  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(this->thread_id_);
    printer << "Resetting prefetch stats..." << std::endl;
  }
  prefetch_manager_->remove_work(this->tree_data_);
  this->tree_data_->wait_for_prefetch_threads(false);

  // TODO: distribute this work across prefetch threads, and/or do something more lightweight
  // than bluntly copying the whole tree.
  this->tree_data_->root_node()->reset_prefetch_stats();
  prefetch_manager_->add_work(this->tree_data_, this->nn_eval_service_, this->search_params_,
                              this->manager_params_);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(this->thread_id_);
    printer << "Prefetch stats reset complete!" << std::endl;
  }
}

}  // namespace mcts
