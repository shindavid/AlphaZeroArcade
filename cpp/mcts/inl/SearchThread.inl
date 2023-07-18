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
  if (thread_ && thread_->joinable()) thread_->join();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::kill() {
  join();
  if (thread_) delete thread_;
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
  return search_active() && stats.count <= tree_size_limit && !root->eliminated();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::visit(Node* tree, int depth) {
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id());
    printer << __func__ << " " << tree->genealogy_str() << " cp=" << (int)tree->stable_data().current_player;
    printer.endl();
  }

  const auto& stable_data = tree->stable_data();
  const auto& outcome = stable_data.outcome;
  if (core::is_terminal_outcome(outcome)) {
    backprop_outcome(tree, outcome);
    perform_eliminations(tree, outcome);
    mark_as_fully_analyzed(tree);
    return;
  }

  if (!search_active()) return;  // short-circuit

  evaluate_and_expand_result_t data = evaluate_and_expand(tree);
  NNEvaluation* evaluation = data.evaluation.get();
  assert(evaluation);

  if (data.backpropagated_virtual_loss) {
    profiler_.record(SearchThreadRegion::kBackpropEvaluation);

    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer(thread_id());
      printer << "backprop_with_virtual_undo " << tree->genealogy_str();
      printer << " " << evaluation->value_prob_distr().transpose();
      printer.endl();
    }

    tree->backprop_with_virtual_undo(evaluation->value_prob_distr());
  } else {
    child_index_t best_child_index = get_best_child_index(tree, evaluation);
    Node* child = tree->init_child(best_child_index);
    visit(child, depth + 1);
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
inline void SearchThread<GameState, Tensorizor>::backprop_outcome(Node* tree, const ValueArray& outcome) {
  profiler_.record(SearchThreadRegion::kBackpropOutcome);
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << tree->genealogy_str() << " " << outcome.transpose();
    printer.endl();
  }

  tree->backprop(outcome);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::perform_eliminations(Node* tree, const ValueArray& outcome) {
  if (manager_params_->disable_eliminations) return;
  player_bitset_t forcibly_winning;
  player_bitset_t forcibly_losing;
  for (int p = 0; p < kNumPlayers; ++p) {
    forcibly_winning.set(p, outcome(p) == 1);
    forcibly_losing.set(p, outcome(p) == 0);
  }
  int cp = tree->stable_data().current_player;
  bool winning = outcome(cp) == 1;
  bool losing = outcome(cp) == 0;
  if (!winning && !losing) return;  // drawn position, no elimination possible

  ValueArray accumulated_value;
  accumulated_value.setZero();
  int accumulated_count = 0;

  profiler_.record(SearchThreadRegion::kPerformEliminations);
  tree->eliminate(thread_id_, forcibly_winning, forcibly_losing, accumulated_value, accumulated_count);
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::mark_as_fully_analyzed(Node* tree) {
  profiler_.record(SearchThreadRegion::kMarkFullyAnalyzed);
  tree->mark_as_fully_analyzed();
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename SearchThread<GameState, Tensorizor>::evaluate_and_expand_result_t
SearchThread<GameState, Tensorizor>::evaluate_and_expand(Node* tree) {
  profiler_.record(SearchThreadRegion::kEvaluateAndExpand);

  std::unique_lock<std::mutex> lock(tree->evaluation_data_mutex());
  typename Node::evaluation_data_t& evaluation_data = tree->evaluation_data();
  evaluate_and_expand_result_t data{evaluation_data.ptr.load(), false};
  auto state = evaluation_data.state;

  switch (state) {
    case Node::kUnset:
    {
      evaluate_and_expand_unset(tree, &lock, &data);
      tree->cv_evaluate_and_expand().notify_all();
      break;
    }
    default: break;
  }
  return data;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void SearchThread<GameState, Tensorizor>::evaluate_and_expand_unset(
    Node* tree, std::unique_lock<std::mutex>* lock, evaluate_and_expand_result_t* data)
{
  profiler_.record(SearchThreadRegion::kEvaluateAndExpandUnset);

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << tree->genealogy_str();
    printer.endl();
  }

  assert(!tree->has_children());
  data->backpropagated_virtual_loss = true;
  assert(data->evaluation.get() == nullptr);

  auto& evaluation_data = tree->evaluation_data();

  profiler_.record(SearchThreadRegion::kVirtualBackprop);
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << "virtual_backprop " << tree->genealogy_str();
    printer.endl();
  }

  tree->virtual_backprop();

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
  if (tree->is_root()) {
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
child_index_t SearchThread<GameState, Tensorizor>::get_best_child_index(Node* tree, NNEvaluation* evaluation) {
  profiler_.record(SearchThreadRegion::kPUCT);

  PUCTStats stats(*manager_params_, *search_params_, tree);

  using PVec = LocalPolicyArray;

  const PVec& P = stats.P;
  const PVec& N = stats.N;
  const PVec& VN = stats.VN;
  const PVec& E = stats.E;
  PVec& PUCT = stats.PUCT;

  bool add_noise = !search_params_->disable_exploration && manager_params_->dirichlet_mult > 0;
  if (manager_params_->forced_playouts && add_noise) {
    PVec n_forced = (P * manager_params_->k_forced * N.sum()).sqrt();
    auto F1 = (N < n_forced).template cast<dtype>();
    auto F2 = (N > 0).template cast<dtype>();
    auto F = F1 * F2;
    PUCT = PUCT * (1 - F) + F * 1e+6;
  }

  PUCT -= E * (100 + PUCT.maxCoeff() - PUCT.minCoeff());  // zero-out where E==1

  int argmax_index;
  PUCT.maxCoeff(&argmax_index);

  if (nn_eval_service_) {
    nn_eval_service_->record_puct_calc(VN.sum() > 0);
  }

  if (mcts::kEnableThreadingDebug) {
    std::string genealogy = tree->genealogy_str();

    util::ThreadSafePrinter printer(thread_id());

    printer << "*************";
    printer.endl();
    printer << __func__ << "() " << genealogy;
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
    printer << "VN: " << stats.VN.transpose();
    printer.endl();
    printer << "E: " << E.transpose();
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
