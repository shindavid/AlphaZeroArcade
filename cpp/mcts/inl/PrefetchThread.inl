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
    threads_.push_back(new PrefetchThread(this, i + 1));
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
PrefetchThread<GameState, Tensorizor>::PrefetchThread(PrefetchThreadManager* manager, int thread_id)
: base_t(kPrefetchMode, manager->profiling_dir(), thread_id)
, manager_(manager)
{
  util::release_assert(thread_id > 0, "PrefetchThread's must use positive thread-id's");
  this->thread_ = new std::thread([this] { loop(); });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void PrefetchThread<GameState, Tensorizor>::loop() {
  while (true) {
    work_item_t work_item;
    if (!manager_->get_next_work_item(&work_item)) return;

    this->search_path_.clear();
    this->shared_data_ = work_item.shared_data;
    this->nn_eval_service_ = work_item.nn_eval_service;
    this->search_params_ = work_item.search_params;
    this->manager_params_ = work_item.manager_params;

    // Incrementing before checking search_active here avoids potential race-condition
    this->shared_data_->increment_active_thread_count();
    if (!this->shared_data_->search_active()) {
      this->shared_data_->decrement_active_thread_count();
      continue;
    }

    Node* root = this->shared_data_->root_node.get();
    prefetch(root, nullptr, this->shared_data_->move_number);
    this->shared_data_->prefetch_notify();

    this->dump_profiling_stats();
    this->shared_data_->decrement_active_thread_count();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void PrefetchThread<GameState, Tensorizor>::prefetch(Node* tree, edge_t* edge,
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

  evaluation_result_t data = evaluate(tree);
  NNEvaluation* evaluation = data.evaluation.get();

  if (data.backpropagated_virtual_loss) {
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer(this->thread_id_);
      printer << "hit leaf node" << std::endl;
    }
    backprop_with_virtual_undo(evaluation->value_prob_distr());
  } else {
    auto& children_data = tree->children_data();
    core::action_index_t action_index = this->get_best_action_index(tree, evaluation);

    edge_t* edge = children_data.find(action_index);
    if (!edge) {
      Action action =
          GameStateTypes::get_nth_valid_action(stable_data.valid_action_mask, action_index);
      auto child = this->shared_data_->node_cache.fetch_or_create(move_number, tree, action);

      std::unique_lock lock(tree->children_mutex());
      edge = children_data.insert(action, action_index, child);
    }

    int edge_count = edge->count(kPrefetchMode);
    int child_count = edge->child()->stats(kPrefetchMode).real_count;
    if (edge_count < child_count) {
      this->short_circuit_backprop(edge);
    } else {
      prefetch(edge->child().get(), edge, move_number + 1);
    }
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void PrefetchThread<GameState, Tensorizor>::virtual_backprop() {
  this->profiler_.record(TreeTraversalThreadRegion::kVirtualBackprop);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(this->thread_id_);
    printer << __func__ << " " << this->search_path_str() << std::endl;
  }

  for (int i = this->search_path_.size() - 1; i >= 0; --i) {
    Node* node = this->search_path_[i].node;
    node->update_stats(VirtualIncrement{}, kPrefetchMode);
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void PrefetchThread<GameState, Tensorizor>::backprop_with_virtual_undo(const ValueArray& value) {
  this->profiler_.record(TreeTraversalThreadRegion::kBackpropWithVirtualUndo);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(this->thread_id_);
    printer << __func__ << " " << this->search_path_str() << " " << value.transpose() << std::endl;
  }

  Node* last_node = this->search_path_.back().node;
  edge_t* last_edge = this->search_path_.back().edge;
  last_node->update_stats(IncrementTransfer{}, kPrefetchMode);
  if (last_edge) last_edge->increment_count(kPrefetchMode);

  for (int i = this->search_path_.size() - 2; i >= 0; --i) {
    Node* child = this->search_path_[i].node;
    edge_t* edge = this->search_path_[i].edge;
    child->update_stats(IncrementTransfer{}, kPrefetchMode);
    if (i) edge->increment_count(kPrefetchMode);
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename PrefetchThread<GameState, Tensorizor>::evaluation_result_t
PrefetchThread<GameState, Tensorizor>::evaluate(Node* tree) {
  this->profiler_.record(TreeTraversalThreadRegion::kEvaluate);

  std::unique_lock<std::mutex> lock(tree->evaluation_data_mutex());
  typename Node::evaluation_data_t& evaluation_data = tree->evaluation_data();
  evaluation_result_t data{evaluation_data.ptr.load(), false};
  auto state = evaluation_data.state;

  switch (state) {
    case Node::kUnset: {
      evaluate_unset(tree, &lock, &data);
      lock.unlock();
      break;
    }
    default:
      break;
  }
  return data;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void PrefetchThread<GameState, Tensorizor>::evaluate_unset(Node* tree,
                                                           std::unique_lock<std::mutex>* lock,
                                                           evaluation_result_t* data) {
  this->profiler_.record(TreeTraversalThreadRegion::kEvaluateUnset);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(this->thread_id_);
    printer << __func__ << " " << this->search_path_str() << std::endl;
  }

  data->backpropagated_virtual_loss = true;
  util::debug_assert(data->evaluation.get() == nullptr);

  auto& evaluation_data = tree->evaluation_data();

  this->virtual_backprop();

  const auto& stable_data = tree->stable_data();
  if (!this->nn_eval_service_) {
    // no-model mode
    ValueTensor uniform_value;
    PolicyTensor uniform_policy;
    uniform_value.setConstant(1.0 / base_t::kNumPlayers);
    uniform_policy.setConstant(0);
    data->evaluation = std::make_shared<NNEvaluation>(uniform_value, uniform_policy,
                                                      stable_data.valid_action_mask);
  } else {
    core::symmetry_index_t sym_index = stable_data.sym_index;
    typename NNEvaluationService::Request request{tree, &this->profiler_, this->thread_id_,
                                                  sym_index};
    auto response = this->nn_eval_service_->evaluate(request);
    data->evaluation = response.ptr;
  }

  LocalPolicyArray P = eigen_util::softmax(data->evaluation->local_policy_logit_distr());
  if (tree == this->shared_data_->root_node.get()) {
    if (!this->search_params_->disable_exploration) {
      if (this->manager_params_->dirichlet_mult) {
        this->add_dirichlet_noise(P);
      }
      P = P.pow(1.0 / this->root_softmax_temperature());
      P /= P.sum();
    }
  }
  evaluation_data.local_policy_prob_distr = P;
  evaluation_data.ptr.store(data->evaluation);
  evaluation_data.state = Node::kSet;
}

}  // namespace mcts
