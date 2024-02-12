#include <mcts/SearchThread.hpp>

#include <util/Asserts.hpp>

#include <boost/algorithm/string.hpp>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline SearchThread<GameState, Tensorizor>::SearchThread(SharedData* shared_data,
                                                         NNEvaluationService* nn_eval_service,
                                                         const ManagerParams* manager_params,
                                                         int thread_id)
    : shared_data_(shared_data),
      nn_eval_service_(nn_eval_service),
      manager_params_(manager_params),
      thread_id_(thread_id) {
  thread_ = new std::thread([=, this] { loop(); });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline SearchThread<GameState, Tensorizor>::~SearchThread() {
  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
  profiler_.dump(1);
  profiler_.close_file();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::set_profiling_dir(
    const boost::filesystem::path& profiling_dir) {
  auto dir = profiling_dir;
  int manager_id = shared_data_->manager_id;
  auto profiling_file_path = dir / util::create_string("search%d-%d.txt", manager_id, thread_id_);
  profiler_.initialize_file(profiling_file_path);
  profiler_.set_name(util::create_string("s-%d-%-2d", manager_id, thread_id_));
  profiler_.skip_next_n_dumps(5);
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::wait_for_activation() const {
  std::unique_lock lock(shared_data_->search_mutex);
  shared_data_->cv_search_on.wait(lock, [this] {
    return shared_data_->shutting_down || shared_data_->active_search_threads[thread_id_];
  });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::perform_visits() {
  Node* root = shared_data_->root_node.get();
  while (root->stats().total_count() <= shared_data_->search_params.tree_size_limit) {
    search_path_.clear();
    state_ = shared_data_->root_state;
    tensorizor_ = shared_data_->root_tensorizor;
    visit(root, nullptr, shared_data_->move_number);
    dump_profiling_stats();
    if (!shared_data_->search_params.ponder && root->stable_data().num_valid_actions == 1) break;
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::deactivate() const {
  std::unique_lock lock(shared_data_->search_mutex);
  shared_data_->active_search_threads[thread_id_] = 0;
  shared_data_->cv_search_off.notify_all();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void SearchThread<GameState, Tensorizor>::loop() {
  while (!shared_data_->shutting_down) {
    wait_for_activation();
    if (shared_data_->shutting_down) break;
    perform_visits();
    deactivate();
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
  if (GameStateTypes::is_terminal_outcome(stable_data.outcome)) {
    pure_backprop(stable_data.outcome);
    return;
  }

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
    bool applied_action = false;
    if (!edge) {
      Action action =
          GameStateTypes::get_nth_valid_action(stable_data.valid_action_mask, action_index);
      GameOutcome outcome = state_.apply_move(action);
      tensorizor_.receive_state_change(state_, action);
      applied_action = true;
      auto child =
          shared_data_->node_cache.fetch_or_create(move_number, state_, outcome, tensorizor_);

      std::unique_lock lock(tree->children_mutex());
      edge = children_data.insert(action, action_index, child);
    }

    int edge_count = edge->count();
    int child_count = edge->child()->stats().real_count;
    if (edge_count < child_count) {
      short_circuit_backprop(edge);
    } else {
      if (!applied_action) {
        state_.apply_move(edge->action());
        tensorizor_.receive_state_change(state_, edge->action());
      }
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
      tree->cv_evaluate().notify_all();
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
    typename NNEvaluationService::Request request{tree,       &state_,    &tensorizor_,
                                                  &profiler_, thread_id_, sym_index};
    auto response = nn_eval_service_->evaluate(request);
    data->evaluation = response.ptr;
  }

  LocalPolicyArray P = eigen_util::softmax(data->evaluation->local_policy_logit_distr());
  if (tree == shared_data_->root_node.get()) {
    if (!shared_data_->search_params.disable_exploration) {
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
    Action action = search_path_[n].edge->action();
    vec.push_back(GameState::action_to_str(action));
  }
  return util::create_string("[%s]", boost::algorithm::join(vec, delim).c_str());
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
core::action_index_t SearchThread<GameState, Tensorizor>::get_best_action_index(
    Node* tree, NNEvaluation* evaluation) {
  profiler_.record(SearchThreadRegion::kPUCT);

  const SearchParams& search_params = shared_data_->search_params;
  PUCTStats stats(*manager_params_, search_params, tree, tree == shared_data_->root_node.get());

  using PVec = LocalPolicyArray;

  const PVec& P = stats.P;
  const PVec& N = stats.N;
  PVec& PUCT = stats.PUCT;

  bool add_noise = !search_params.disable_exploration && manager_params_->dirichlet_mult > 0;
  if (manager_params_->forced_playouts && add_noise) {
    PVec n_forced = (P * manager_params_->k_forced * N.sum()).sqrt();
    auto F1 = (N < n_forced).template cast<dtype>();
    auto F2 = (N > 0).template cast<dtype>();
    auto F = F1 * F2;
    PUCT = PUCT * (1 - F) + F * 1e+6;
  }

  int argmax_index;
  PUCT.maxCoeff(&argmax_index);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id());

    const auto& tree_stats = tree->stats();
    core::seat_index_t cp = tree->stable_data().current_player;

    using ArrayT1 = Eigen::Array<dtype, 3, kNumPlayers>;
    ArrayT1 A1;
    A1.setZero();
    A1.row(0) = tree_stats.real_avg;
    A1.row(1) = tree_stats.virtualized_avg;
    A1(2, cp) = 1;

    std::ostringstream ss1;
    ss1 << A1;
    std::string s1 = ss1.str();

    std::vector<std::string> s1_lines;
    boost::split(s1_lines, s1, boost::is_any_of("\n"));

    printer << "real_avg: " << s1_lines[0] << std::endl;
    printer << "virt_avg: " << s1_lines[1] << std::endl;

    std::string cp_line = s1_lines[2];
    std::replace(cp_line.begin(), cp_line.end(), '0', ' ');
    std::replace(cp_line.begin(), cp_line.end(), '1', '*');

    printer << "cp:       " << cp_line << std::endl;

    using ScalarT = typename PVec::Scalar;
    constexpr int kNumRows1 = PolicyShape::count;
    constexpr int kNumRows2 = 9;  // P, V, PW, PL, E, N, VN, PUCT, argmax
    constexpr int kNumRows = kNumRows1 + kNumRows2;
    constexpr int kMaxCols = PVec::MaxRowsAtCompileTime;
    using ArrayT2 = Eigen::Array<ScalarT, kNumRows, Eigen::Dynamic, 0, kNumRows, kMaxCols>;

    ArrayT2 A2(kNumRows, P.rows());
    A2.setZero();

    const ActionMask& valid_actions = tree->stable_data().valid_action_mask;
    const bool* data = valid_actions.data();
    int r = 0;
    int c = 0;
    for (int i = 0; i < kNumGlobalActionsBound; ++i) {
      if (!data[i]) continue;
      Action action = eigen_util::unflatten_index(valid_actions, i);
      for (int j = 0; j < kNumRows1; ++j) {
        A2(j, c) = action[j];
      }
      ++c;
    }

    r += kNumRows1;

    A2.row(r++) = P;
    A2.row(r++) = stats.V;
    A2.row(r++) = stats.PW;
    A2.row(r++) = stats.PL;
    A2.row(r++) = stats.E;
    A2.row(r++) = N;
    A2.row(r++) = stats.VN;
    A2.row(r++) = PUCT;
    A2(r, argmax_index) = 1;

    std::ostringstream ss2;
    ss2 << A2;
    std::string s2 = ss2.str();

    std::vector<std::string> s2_lines;
    boost::split(s2_lines, s2, boost::is_any_of("\n"));

    r = 0;
    printer << "move:  " << s2_lines[r++] << std::endl;
    while (r < kNumRows1) {
      printer << "       " << s2_lines[r++] << std::endl;
    }
    printer << "P:     " << s2_lines[r++] << std::endl;
    printer << "V:     " << s2_lines[r++] << std::endl;
    printer << "PW:    " << s2_lines[r++] << std::endl;
    printer << "PL:    " << s2_lines[r++] << std::endl;
    printer << "E:     " << s2_lines[r++] << std::endl;
    printer << "N:     " << s2_lines[r++] << std::endl;
    printer << "VN:    " << s2_lines[r++] << std::endl;
    printer << "PUCT:  " << s2_lines[r++] << std::endl;

    std::string argmax_line = s2_lines[r];
    std::replace(argmax_line.begin(), argmax_line.end(), '0', ' ');
    std::replace(argmax_line.begin(), argmax_line.end(), '1', '*');

    printer << "argmax:" << argmax_line << std::endl;
    printer << "*************" << std::endl;
  }
  return argmax_index;
}

}  // namespace mcts
