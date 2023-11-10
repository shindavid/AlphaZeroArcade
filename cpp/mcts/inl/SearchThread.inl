#include <mcts/SearchThread.hpp>

#include <util/Asserts.hpp>

#include <boost/algorithm/string.hpp>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
SearchThreadManager<GameState, Tensorizor>* SearchThreadManager<GameState, Tensorizor>::instance_ =
    nullptr;

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
SearchThreadManager<GameState, Tensorizor>* SearchThreadManager<GameState, Tensorizor>::get(
    const ManagerParams& params) {
  if (!instance_) {
    instance_ = new SearchThreadManager(params.profiling_dir());
  } else {
    util::release_assert(params.profiling_dir() == instance_->profiling_dir_,
                         "Inconsistent profiling dirs (%s vs %s)", params.profiling_dir().c_str(),
                         instance_->profiling_dir_.c_str());
  }

  instance_->add_threads_if_necessary(params.num_search_threads);
  return instance_;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThreadManager<GameState, Tensorizor>::shutdown() {
  if (shutdown_initiated_) return;

  std::cout << "SearchThreadManager::shutdown()" << std::endl;
  shutdown_initiated_ = true;
  work_items_cv_.notify_all();
  for (auto thread : threads_) {
    delete thread;
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThreadManager<GameState, Tensorizor>::add_work(SharedData* shared_data,
                                                          NNEvaluationService* nn_eval_service,
                                                          const SearchParams* search_params,
                                                          const ManagerParams* manager_params) {
  shared_data->seeking_search_threads = true;
  std::unique_lock lock(work_items_mutex_);
  work_items_.emplace_back(shared_data, nn_eval_service, search_params, manager_params);
  work_item_index_ = work_items_.size() - 1;
  lock.unlock();
  work_items_cv_.notify_all();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThreadManager<GameState, Tensorizor>::remove_work(SharedData* shared_data) {
  shared_data->seeking_search_threads = false;

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
void SearchThreadManager<GameState, Tensorizor>::wait_for_completion(SharedData* shared_data) {
  shared_data->wait_for_search_completion();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
bool SearchThreadManager<GameState, Tensorizor>::get_next_work_item(work_item_t* work_item) {
  std::unique_lock lock(work_items_mutex_);

  work_items_cv_.wait(lock, [&] { return get_next_work_item_helper(work_item); });

  return !shutdown_initiated_;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
bool SearchThreadManager<GameState, Tensorizor>::get_next_work_item_helper(work_item_t* work_item) {
  if (shutdown_initiated_) return true;
  if (work_items_.empty()) return false;
  if (work_item_index_ >= (int)work_items_.size()) work_item_index_ = 0;
  *work_item = work_items_[work_item_index_++];
  return true;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThreadManager<GameState, Tensorizor>::add_threads_if_necessary(int num_total_threads) {
  int num_cur_threads = threads_.size();
  for (int i = num_cur_threads; i < num_total_threads; ++i) {
    threads_.push_back(new SearchThread(this, i));
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline SearchThread<GameState, Tensorizor>::SearchThread(SearchThreadManager* manager,
                                                         int thread_id)
    : manager_(manager), thread_id_(thread_id) {
  if (kEnableProfiling) {
    auto dir = manager->profiling_dir();
    auto profiling_file_path = dir / util::create_string("search-%d.txt", thread_id);
    profiler_.initialize_file(profiling_file_path);
    profiler_.set_name(util::create_string("s-%-3d", thread_id));
    profiler_.skip_next_n_dumps(5);
  }

  thread_ = new std::thread([this] { loop(); });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline SearchThread<GameState, Tensorizor>::~SearchThread() {
  kill();
  profiler_.dump(1);
  profiler_.close_file();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::join() {
  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::kill() {
  join();
  if (thread_) {
    delete thread_;
    thread_ = nullptr;
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::loop() {
  while (true) {
    if (shared_data_) {
      shared_data_->increment_active_search_thread_count(-1);
    }
    work_item_t work_item;
    if (!manager_->get_next_work_item(&work_item)) return;

    search_path_.clear();
    shared_data_ = work_item.shared_data;
    nn_eval_service_ = work_item.nn_eval_service;
    search_params_ = work_item.search_params;
    manager_params_ = work_item.manager_params;

    shared_data_->increment_active_search_thread_count(+1);
    if (!shared_data_->seeking_search_threads) {
      continue;
    }

    Node* root = shared_data_->root_node.get();
    visit(root, nullptr, shared_data_->move_number);

    if (root->stats().total_count() > search_params_->tree_size_limit) {
      manager_->remove_work(shared_data_);
    }

    dump_profiling_stats();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::visit(Node* tree, edge_t* edge,
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
  if (core::is_terminal_outcome(outcome)) {
    pure_backprop(outcome);
    return;
  }

  if (!shared_data_->seeking_search_threads) return;  // short-circuit

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
    int child_count = edge->child()->stats().real_count;
    if (edge_count < child_count) {
      short_circuit_backprop(edge);
    } else {
      visit(edge->child().get(), edge, move_number + 1);
    }
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::add_dirichlet_noise(LocalPolicyArray& P) {
  int rows = P.rows();
  double alpha = manager_params_->dirichlet_alpha_factor / sqrt(rows);
  LocalPolicyArray noise = dirichlet_gen().template generate<LocalPolicyArray>(rng(), alpha, rows);
  P = (1.0 - manager_params_->dirichlet_mult) * P + manager_params_->dirichlet_mult * noise;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::virtual_backprop() {
  profiler_.record(SearchThreadRegion::kVirtualBackprop);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << std::endl;
  }

  for (int i = search_path_.size() - 1; i >= 0; --i) {
    Node* node = search_path_[i].node;
    node->update_stats(VirtualIncrement{});
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::pure_backprop(const ValueArray& value) {
  profiler_.record(SearchThreadRegion::kPureBackprop);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << " " << value.transpose() << std::endl;
  }

  Node* last_node = search_path_.back().node;
  edge_t* last_edge = search_path_.back().edge;
  last_node->update_stats(SetEvalExact(value));
  last_edge->increment_count();

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    Node* child = search_path_[i].node;
    edge_t* edge = search_path_[i].edge;
    child->update_stats(RealIncrement{});
    if (i) edge->increment_count();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::backprop_with_virtual_undo(const ValueArray& value) {
  profiler_.record(SearchThreadRegion::kBackpropWithVirtualUndo);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << " " << value.transpose() << std::endl;
  }

  Node* last_node = search_path_.back().node;
  edge_t* last_edge = search_path_.back().edge;
  last_node->update_stats(SetEvalWithVirtualUndo(value));
  if (last_edge) last_edge->increment_count();

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    Node* child = search_path_[i].node;
    edge_t* edge = search_path_[i].edge;
    child->update_stats(IncrementTransfer{});
    if (i) edge->increment_count();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::short_circuit_backprop(edge_t* last_edge) {
  // short-circuit
  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << std::endl;
  }

  last_edge->increment_count();

  for (int i = search_path_.size() - 1; i >= 0; --i) {
    Node* child = search_path_[i].node;
    edge_t* edge = search_path_[i].edge;
    child->update_stats(RealIncrement{});
    if (i) edge->increment_count();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename SearchThread<GameState, Tensorizor>::evaluation_result_t
SearchThread<GameState, Tensorizor>::evaluate(Node* tree) {
  profiler_.record(SearchThreadRegion::kEvaluate);

  std::unique_lock<std::mutex> lock(tree->evaluation_data_mutex());
  typename Node::evaluation_data_t& evaluation_data = tree->evaluation_data();
  evaluation_result_t data{evaluation_data.ptr.load(), false};
  auto state = evaluation_data.state;

  switch (state) {
    case Node::kUnset: {
      evaluate_unset(tree, &lock, &data);
      break;
    }
    default:
      break;
  }
  return data;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::evaluate_unset(Node* tree,
                                                         std::unique_lock<std::mutex>* lock,
                                                         evaluation_result_t* data) {
  profiler_.record(SearchThreadRegion::kEvaluateUnset);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << std::endl;
  }

  data->backpropagated_virtual_loss = true;
  util::debug_assert(data->evaluation.get() == nullptr);

  auto& evaluation_data = tree->evaluation_data();

  virtual_backprop();

  const auto& stable_data = tree->stable_data();
  if (!nn_eval_service_) {
    // no-model mode
    ValueTensor uniform_value;
    PolicyTensor uniform_policy;
    uniform_value.setConstant(1.0 / kNumPlayers);
    uniform_policy.setConstant(0);
    data->evaluation = std::make_shared<NNEvaluation>(uniform_value, uniform_policy,
                                                      stable_data.valid_action_mask);
  } else {
    core::symmetry_index_t sym_index = stable_data.sym_index;
    typename NNEvaluationService::Request request{tree, &profiler_, thread_id_, sym_index};
    auto response = nn_eval_service_->evaluate(request);
    data->evaluation = response.ptr;
  }

  LocalPolicyArray P = eigen_util::softmax(data->evaluation->local_policy_logit_distr());
  if (tree == shared_data_->root_node.get()) {
    if (!search_params_->disable_exploration) {
      if (manager_params_->dirichlet_mult) {
        add_dirichlet_noise(P);
      }
      P = P.pow(1.0 / root_softmax_temperature());
      P /= P.sum();
    }
  }
  evaluation_data.local_policy_prob_distr = P;
  evaluation_data.ptr.store(data->evaluation);
  evaluation_data.state = Node::kSet;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
std::string SearchThread<GameState, Tensorizor>::search_path_str() const {
  std::string delim = GameState::action_delimiter();
  std::vector<std::string> vec;
  for (int n = 1; n < (int)search_path_.size(); ++n) {  // skip the first node
    const GameState& prev_state = search_path_[n - 1].node->stable_data().state;
    Action action = search_path_[n].edge->action();
    vec.push_back(prev_state.action_to_str(action));
  }
  return util::create_string("[%s]", boost::algorithm::join(vec, delim).c_str());
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
core::action_index_t SearchThread<GameState, Tensorizor>::get_best_action_index(
    Node* tree, NNEvaluation* evaluation) {
  profiler_.record(SearchThreadRegion::kPUCT);

  PUCTStats stats(*manager_params_, *search_params_, tree, tree == shared_data_->root_node.get());

  using PVec = LocalPolicyArray;

  const PVec& P = stats.P;
  const PVec& N = stats.N;
  const PVec& VN = stats.VN;
  PVec& PUCT = stats.PUCT;

  bool add_noise = !search_params_->disable_exploration && manager_params_->dirichlet_mult > 0;
  if (manager_params_->forced_playouts && add_noise) {
    PVec n_forced = (P * manager_params_->k_forced * N.sum()).sqrt();
    auto F1 = (N < n_forced).template cast<dtype>();
    auto F2 = (N > 0).template cast<dtype>();
    auto F = F1 * F2;
    PUCT = PUCT * (1 - F) + F * 1e+6;
  }

  int argmax_index;
  PUCT.maxCoeff(&argmax_index);

  if (nn_eval_service_) {
    nn_eval_service_->record_puct_calc(VN.sum() > 0);
  }

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id());

    printer << "*************" << std::endl;
    printer << __func__ << "() " << search_path_str() << std::endl;
    printer << "real_avg: " << tree->stats().real_avg.transpose() << std::endl;
    printer << "virt_avg: " << tree->stats().virtualized_avg.transpose() << std::endl;

    using ScalarT = typename PVec::Scalar;
    constexpr int kNumCols1 = PolicyShape::count;
    constexpr int kNumCols2 = 8;
    using ArrayT1 = Eigen::Array<ScalarT, Eigen::Dynamic, kNumCols1, 0, PVec::MaxRowsAtCompileTime>;
    using ArrayT2 = Eigen::Array<ScalarT, Eigen::Dynamic, kNumCols2, 0, PVec::MaxRowsAtCompileTime>;

    ArrayT1 A(P.rows(), kNumCols1);
    const ActionMask& valid_actions = tree->stable_data().valid_action_mask;
    const bool* data = valid_actions.data();
    int c = 0;
    for (int i = 0; i < kNumGlobalActionsBound; ++i) {
      if (!data[i]) continue;
      Action action = eigen_util::unflatten_index(valid_actions, i);
      for (int j = 0; j < kNumCols1; ++j) {
        A(c, j) = action[j];
      }
    }

    std::ostringstream ss1;
    ss1 << A.transpose();
    std::string s1 = ss1.str();

    std::vector<std::string> s1_lines;
    boost::split(s1_lines, s1, boost::is_any_of("\n"));

    printer << "valid: " << s1_lines[0] << std::endl;
    for (int i = 1; i < kNumCols1; ++i) {
      printer << "       " << s1_lines[i] << std::endl;
    }

    ArrayT2 M(P.rows(), kNumCols2);

    M.col(0) = P;
    M.col(1) = stats.V;
    M.col(2) = stats.PW;
    M.col(3) = stats.PL;
    M.col(4) = stats.E;
    M.col(5) = N;
    M.col(6) = stats.VN;
    M.col(7) = PUCT;

    std::ostringstream ss2;
    ss2 << M.transpose();
    std::string s2 = ss2.str();

    std::vector<std::string> s2_lines;
    boost::split(s2_lines, s2, boost::is_any_of("\n"));

    printer << "P:     " << s2_lines[0] << std::endl;
    printer << "V:     " << s2_lines[1] << std::endl;
    printer << "PW:    " << s2_lines[2] << std::endl;
    printer << "PL:    " << s2_lines[3] << std::endl;
    printer << "E:     " << s2_lines[4] << std::endl;
    printer << "N:     " << s2_lines[5] << std::endl;
    printer << "VN:    " << s2_lines[6] << std::endl;
    printer << "PUCT:  " << s2_lines[7] << std::endl;

    printer << "argmax: " << argmax_index << std::endl;
    printer << "*************" << std::endl;
  }
  return argmax_index;
}

}  // namespace mcts
