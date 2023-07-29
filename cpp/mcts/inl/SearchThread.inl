#include <mcts/SearchThread.hpp>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline SearchThread<GameState, Tensorizor>::SearchThread(
    SharedData* shared_data, NNEvaluationService* nn_eval_service, const ManagerParams* manager_params, int thread_id)
: shared_data_(shared_data)
, nn_eval_service_(nn_eval_service)
, manager_params_(manager_params)
, thread_id_(thread_id) {
  if (kEnableProfiling) {
    auto dir = manager_params->profiling_dir();
    int manager_id = shared_data->manager_id;
    auto profiling_file_path = dir / util::create_string("search%d-%d.txt", manager_id, thread_id);
    profiler_.initialize_file(profiling_file_path);
    profiler_.set_name(util::create_string("s-%d-%-2d", manager_id, thread_id));
    profiler_.skip_next_n_dumps(5);
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline SearchThread<GameState, Tensorizor>::~SearchThread() {
  kill();
  profiler_.dump(1);
  profiler_.close_file();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::join() {
  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::kill() {
  join();
  if (thread_) {
    delete thread_;
    thread_ = nullptr;
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::launch(const SearchParams* search_params, std::function<void()> f) {
  kill();
  search_params_ = search_params;
  thread_ = new std::thread(f);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
bool SearchThread<GameState, Tensorizor>::needs_more_visits(Node* root, int tree_size_limit) {
  profiler_.record(SearchThreadRegion::kCheckVisitReady);
  const auto& stats = root->stats();
  return search_active() && stats.count <= tree_size_limit;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::run() {
  search_path_.clear();
  visit(shared_data_->root_node, -1, shared_data_->move_number);
  dump_profiling_stats();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::visit(Node* tree, child_index_t child_index, move_number_t move_number) {
  search_path_.emplace_back(tree, child_index);

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id());
    printer << __func__ << "(" << child_index << ") " << search_path_str() << " cp=" << (int)tree->stable_data().current_player;
    printer.endl();
  }

  const auto& stable_data = tree->stable_data();
  const auto& outcome = stable_data.outcome;
  if (core::is_terminal_outcome(outcome)) {
    backprop(outcome);
    return;
  }

  if (!search_active()) return;  // short-circuit

  evaluation_result_t data = evaluate(tree);
  NNEvaluation* evaluation = data.evaluation.get();
  assert(evaluation);

  if (data.backpropagated_virtual_loss) {
    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer(thread_id());
      printer << "hit leaf node";
      printer.endl();
    }
    backprop_with_virtual_undo(evaluation->value_prob_distr());
  } else {
    child_index_t best_child_index = get_best_child_index(tree, evaluation);

    traverse_request_t request{&shared_data_->node_cache, best_child_index, shared_data_->move_number, .01};
    ValueArray backprop_value = tree->traverse(request);
    Node* child = tree->get_child(best_child_index);
    if (backprop_value.sum() == 0) {
      visit(child, best_child_index, move_number + 1);
    } else {
      search_path_.emplace_back(child, best_child_index);
      if (mcts::kEnableThreadingDebug) {
        util::ThreadSafePrinter printer(thread_id());
        printer << "hit transposition node with big delta: " << backprop_value.transpose();
        printer.endl();
      }
      backprop(backprop_value);
    }
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::add_dirichlet_noise(LocalPolicyArray& P) {
  int rows = P.rows();
  double alpha = manager_params_->dirichlet_alpha_sum / rows;
  LocalPolicyArray noise = dirichlet_gen().template generate<LocalPolicyArray>(rng(), alpha, rows);
  P = (1.0 - manager_params_->dirichlet_mult) * P + manager_params_->dirichlet_mult * noise;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::virtual_backprop() {
  profiler_.record(SearchThreadRegion::kVirtualBackprop);

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str();
    printer.endl();
  }

  for (int i = search_path_.size() - 1; i >= 1; --i) {
    Node* child = search_path_[i].node;
    Node* parent = search_path_[i-1].node;
    child_index_t child_index = search_path_[i].child_index;
    child->virtual_backprop(parent, child_index);
  }
  search_path_[0].node->virtual_backprop();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::backprop(const ValueArray& value) {
  profiler_.record(SearchThreadRegion::kBackprop);

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << " " << value.transpose();
    printer.endl();
  }

  ValueArray value_copy = value;
  for (int i = search_path_.size() - 1; i >= 1; --i) {
    Node* child = search_path_[i].node;
    Node* parent = search_path_[i-1].node;
    child_index_t child_index = search_path_[i].child_index;
    child->backprop(value_copy, parent, child_index);
  }
  search_path_[0].node->backprop(value_copy);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::backprop_with_virtual_undo(const ValueArray& value) {
  profiler_.record(SearchThreadRegion::kBackpropWithVirtualUndo);

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << " " << value.transpose();
    printer.endl();
  }

  ValueArray value_copy = value;
  for (int i = search_path_.size() - 1; i >= 1; --i) {
    Node* child = search_path_[i].node;
    Node* parent = search_path_[i-1].node;
    child_index_t child_index = search_path_[i].child_index;
    child->backprop_with_virtual_undo(value_copy, parent, child_index);
  }
  search_path_[0].node->backprop_with_virtual_undo(value_copy);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename SearchThread<GameState, Tensorizor>::evaluation_result_t
SearchThread<GameState, Tensorizor>::evaluate(Node* tree) {
  profiler_.record(SearchThreadRegion::kEvaluate);

  std::unique_lock<std::mutex> lock(tree->evaluation_data_mutex());
  typename Node::evaluation_data_t& evaluation_data = tree->evaluation_data();
  evaluation_result_t data{evaluation_data.ptr.load(), false};
  auto state = evaluation_data.state;

  switch (state) {
    case Node::kUnset:
    {
      evaluate_unset(tree, &lock, &data);
      tree->cv_evaluate().notify_all();
      break;
    }
    default: break;
  }
  return data;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::evaluate_unset(
    Node* tree, std::unique_lock<std::mutex>* lock, evaluation_result_t* data)
{
  profiler_.record(SearchThreadRegion::kEvaluateUnset);

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str();
    printer.endl();
  }

  data->backpropagated_virtual_loss = true;
  assert(data->evaluation.get() == nullptr);

  auto& evaluation_data = tree->evaluation_data();

  virtual_backprop();

  const auto& stable_data = tree->stable_data();
  if (!nn_eval_service_) {
    // no-model mode
    ValueTensor uniform_value;
    PolicyTensor uniform_policy;
    uniform_value.setConstant(1.0 / kNumPlayers);
    uniform_policy.setConstant(0);
    data->evaluation = std::make_shared<NNEvaluation>(uniform_value, uniform_policy, stable_data.valid_action_mask);
  } else {
    core::symmetry_index_t sym_index = stable_data.sym_index;
    typename NNEvaluationService::Request request{tree, &profiler_, thread_id_, sym_index};
    auto response = nn_eval_service_->evaluate(request);
    data->evaluation = response.ptr;
  }

  LocalPolicyArray P = eigen_util::softmax(data->evaluation->local_policy_logit_distr());
  if (tree == shared_data_->root_node) {
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

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
std::string SearchThread<GameState, Tensorizor>::search_path_str() const {
  const char* delim = kNumGlobalActions < 10 ? "" : ":";
  std::vector<std::string> vec;
  for (int n = 1; n < (int)search_path_.size(); ++n) {  // skip the first node
    Node* parent = search_path_[n-1].node;
    child_index_t c = search_path_[n].child_index;
    core::action_index_t action = parent->lookup_action_by_child_index(c);
    vec.push_back(std::to_string(action));
  }
  return util::create_string("[%s]", boost::algorithm::join(vec, delim).c_str());
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
child_index_t SearchThread<GameState, Tensorizor>::get_best_child_index(Node* tree, NNEvaluation* evaluation) {
  profiler_.record(SearchThreadRegion::kPUCT);

  PUCTStats stats(*manager_params_, *search_params_, tree, tree == shared_data_->root_node);

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

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id());

    printer << "*************";
    printer.endl();
    printer << __func__ << "() " << search_path_str();
    printer.endl();
    printer << "valid:";
    for (int v : bitset_util::on_indices(tree->stable_data().valid_action_mask)) {
      printer << " " << v;
    }
    printer.endl();
    printer << "value_avg: " << tree->stats().value_avg.transpose();
    printer.endl();
    printer << "P: " << P.transpose();
    printer.endl();
    printer << "N: " << N.transpose();
    printer.endl();
    printer << "V: " << stats.V.transpose();
    printer.endl();
    printer << "E: " << stats.E.transpose();
    printer.endl();
    printer << "VN: " << stats.VN.transpose();
    printer.endl();
    printer << "PUCT: " << PUCT.transpose();
    printer.endl();
    printer << "argmax: " << argmax_index;
    printer.endl();
    printer << "*************";
    printer.endl();
  }
  return argmax_index;
}

}  // namespace mcts
