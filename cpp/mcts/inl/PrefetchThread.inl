#include <mcts/PrefetchThread.hpp>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
PrefetchThreadManager<GameState, Tensorizor>*
    PrefetchThreadManager<GameState, Tensorizor>::instance_ = nullptr;

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
PrefetchThreadManager<GameState, Tensorizor>* PrefetchThreadManager<GameState, Tensorizor>::get(
    const ManagerParams& params) {
  if (!instance_) {
    instance_ = new PrefetchThreadManager(params.profiling_dir());
  } else {
    util::release_assert(params.profiling_dir() == instance_->profiling_dir_,
                         "Inconsistent profiling dirs (%s vs %s)", params.profiling_dir().c_str(),
                         instance_->profiling_dir_.c_str());
  }

  instance_->add_threads_if_necessary(params.num_search_threads);
  return instance_;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void PrefetchThreadManager<GameState, Tensorizor>::shutdown() {
  if (shutdown_initiated_) return;

  std::cout << "PrefetchThreadManager::shutdown()" << std::endl;
  shutdown_initiated_ = true;
  work_items_cv_.notify_all();
  for (auto thread : threads_) {
    delete thread;
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void PrefetchThreadManager<GameState, Tensorizor>::add_work(SharedData* shared_data,
                                                            NNEvaluationService* nn_eval_service,
                                                            const SearchParams* search_params,
                                                            const ManagerParams* manager_params) {
  std::unique_lock lock(work_items_mutex_);
  work_items_.emplace_back(shared_data, nn_eval_service, search_params, manager_params);
  work_item_index_ = work_items_.size() - 1;
  lock.unlock();
  work_items_cv_.notify_all();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void PrefetchThreadManager<GameState, Tensorizor>::remove_work(SharedData* shared_data) {
  shared_data->deactivate_search();

  std::unique_lock lock(work_items_mutex_);
  auto it = work_items_.begin();
  while (it != work_items_.end()) {
    if (it->shared_data == shared_data) {
      work_items_.erase(it);
    } else {
      ++it;
    }
  }

  lock.unlock();
  work_items_cv_.notify_all();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void PrefetchThreadManager<GameState, Tensorizor>::wait_for_completion(SharedData* shared_data) {
  shared_data->wait_for_search_completion();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
bool PrefetchThreadManager<GameState, Tensorizor>::get_next_work_item(work_item_t* work_item) {
  std::unique_lock lock(work_items_mutex_);

  work_items_cv_.wait(lock, [&] { return get_next_work_item_helper(work_item); });

  return !shutdown_initiated_;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
bool PrefetchThreadManager<GameState, Tensorizor>::get_next_work_item_helper(
    work_item_t* work_item) {
  if (shutdown_initiated_) return true;
  if (work_items_.empty()) return false;
  if (work_item_index_ >= (int)work_items_.size()) work_item_index_ = 0;
  *work_item = work_items_[work_item_index_++];
  return true;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void PrefetchThreadManager<GameState, Tensorizor>::add_threads_if_necessary(int num_total_threads) {
  int num_cur_threads = threads_.size();
  for (int i = num_cur_threads; i < num_total_threads; ++i) {
    threads_.push_back(new PrefetchThread(this, i));
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
PrefetchThread<GameState, Tensorizor>::PrefetchThread(PrefetchManager* manager, int thread_id)
: base_t(kPrefetchMode, manager->profiling_dir(), thread_id)
, manager_(manager)
{
  util::release_assert(thread_id > 0, "PrefetchThread's must use positive thread-id's");
  thread_ = new std::thread([this] { loop(); });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void PrefetchThread<GameState, Tensorizor>::loop() {
  while (true) {
    work_item_t work_item;
    if (!manager_->get_next_work_item(&work_item)) return;

    search_path_.clear();
    shared_data_ = work_item.shared_data;
    nn_eval_service_ = work_item.nn_eval_service;
    search_params_ = work_item.search_params;
    manager_params_ = work_item.manager_params;

    shared_data_->increment_active_thread_count();
    if (!shared_data_->search_active()) {
      shared_data_->decrement_active_thread_count();
      continue;
    }

    Node* root = shared_data_->root_node.get();
    prefetch(root, nullptr, shared_data_->move_number);

    if (root->stats().total_count() > search_params_->tree_size_limit) {
      manager_->remove_work(shared_data_);
    }

    dump_profiling_stats();
    shared_data_->decrement_active_thread_count();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void PrefetchThread<GameState, Tensorizor>::prefetch(Node* tree, edge_t* edge,
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
      prefetch(edge->child().get(), edge, move_number + 1);
    }
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void PrefetchThread<GameState, Tensorizor>::virtual_backprop() {
  profiler_.record(TreeTraversalThreadRegion::kVirtualBackprop);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << std::endl;
  }

  for (int i = search_path_.size() - 1; i >= 0; --i) {
    Node* node = search_path_[i].node;
    node->update_stats(VirtualIncrement{}, traversal_mode_);
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void PrefetchThread<GameState, Tensorizor>::backprop_with_virtual_undo(const ValueArray& value) {
  profiler_.record(TreeTraversalThreadRegion::kBackpropWithVirtualUndo);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << " " << value.transpose() << std::endl;
  }

  Node* last_node = search_path_.back().node;
  edge_t* last_edge = search_path_.back().edge;
  last_node->update_stats(IncrementTransfer{}, traversal_mode_);
  if (last_edge) last_edge->increment_count(traversal_mode_);

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    Node* child = search_path_[i].node;
    edge_t* edge = search_path_[i].edge;
    child->update_stats(IncrementTransfer{}, traversal_mode_);
    if (i) edge->increment_count(traversal_mode_);
  }
}

}  // namespace mcts
