#include <mcts/SearchThread.hpp>

#include <util/Asserts.hpp>

#include <boost/algorithm/string.hpp>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace mcts {

template <core::concepts::Game Game>
inline SearchThread<Game>::SearchThread(SharedData* shared_data,
                                        NNEvaluationService* nn_eval_service,
                                        const ManagerParams* manager_params,
                                        int thread_id)
    : shared_data_(shared_data),
      nn_eval_service_(nn_eval_service),
      manager_params_(manager_params),
      thread_id_(thread_id) {
  thread_ = new std::thread([=, this] { loop(); });
}

template <core::concepts::Game Game>
inline SearchThread<Game>::~SearchThread() {
  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
  profiler_.dump(1);
  profiler_.close_file();
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::set_profiling_dir(const boost::filesystem::path& profiling_dir) {
  auto dir = profiling_dir;
  int manager_id = shared_data_->manager_id;
  auto profiling_file_path = dir / util::create_string("search%d-%d.txt", manager_id, thread_id_);
  profiler_.initialize_file(profiling_file_path);
  profiler_.set_name(util::create_string("s-%d-%-2d", manager_id, thread_id_));
  profiler_.skip_next_n_dumps(5);
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::wait_for_activation() const {
  std::unique_lock lock(shared_data_->search_mutex);
  shared_data_->cv_search_on.wait(lock, [this] {
    return shared_data_->shutting_down || shared_data_->active_search_threads[thread_id_];
  });
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::perform_visits() {
  Node* root = shared_data_->root_node.get();
  while (root->stats().total_count() <= shared_data_->search_params.tree_size_limit) {
    search_path_.clear();
    state_ = shared_data_->root_state;
    state_history_ = shared_data_->root_state_history;
    visit(root, nullptr, shared_data_->move_number);
    dump_profiling_stats();
    if (!shared_data_->search_params.ponder && root->stable_data().num_valid_actions == 1) break;
  }
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::deactivate() const {
  std::unique_lock lock(shared_data_->search_mutex);
  shared_data_->active_search_threads[thread_id_] = 0;
  shared_data_->cv_search_off.notify_all();
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::loop() {
  while (!shared_data_->shutting_down) {
    wait_for_activation();
    if (shared_data_->shutting_down) break;
    perform_visits();
    deactivate();
  }
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::visit(Node* node, edge_t* edge, move_number_t move_number) {
  search_path_.emplace_back(node, edge);

  if (mcts::kEnableDebug) {
    std::ostringstream ss;
    ss << thread_id_whitespace();
    if (edge) {
      ss << __func__ << " " << edge->action();
    } else {
      ss << __func__ << "()";
    }
    ss << " " << search_path_str() << " cp=" << (int)node->stable_data().current_player;
    LOG_INFO << ss.str();
  }

  const auto& stable_data = node->stable_data();
  if (stable_data.outcome.terminal) {
    pure_backprop(stable_data.outcome.terminal_value);
    return;
  }

  evaluation_result_t data = evaluate(node);
  NNEvaluation* evaluation = data.evaluation.get();

  if (data.backpropagated_virtual_loss) {
    if (mcts::kEnableDebug) {
      LOG_INFO << thread_id_whitespace() << "hit leaf node";
    }
    backprop_with_virtual_undo(evaluation->value_prob_distr());
  } else {
    auto& children_data = node->children_data();
    core::action_index_t action_index = get_best_action_index(node, evaluation);

    edge_t* edge = children_data.find(action_index);
    bool applied_action = false;
    if (!edge) {
      core::action_t action = bitset_util::get_nth_on_index(stable_data.valid_action_mask,
                                                            action_index);
      core::ActionOutcome outcome = Rules::apply(state_, action);
      util::stuff_back<Game::kHistorySize>(state_history_, state_.base());
      applied_action = true;
      auto child = shared_data_->node_cache.fetch_or_create(move_number, state_, outcome,
                                                            this->manager_params_);

      std::unique_lock lock(node->children_mutex());
      edge = children_data.insert(action, action_index, child);
    }

    int edge_count = edge->count();
    int child_count = edge->child()->stats().real_count;
    if (edge_count < child_count) {
      short_circuit_backprop(edge);
    } else {
      if (!applied_action) {
        Rules::apply(state_, edge->action());
        util::stuff_back<Game::kHistorySize>(state_history_, state_.base());
      }
      visit(edge->child().get(), edge, move_number + 1);
    }
  }
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::add_dirichlet_noise(LocalPolicyArray& P) {
  int rows = P.rows();
  double alpha = manager_params_->dirichlet_alpha_factor / sqrt(rows);
  LocalPolicyArray noise = dirichlet_gen().template generate<LocalPolicyArray>(rng(), alpha, rows);
  P = (1.0 - manager_params_->dirichlet_mult) * P + manager_params_->dirichlet_mult * noise;
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::virtual_backprop() {
  profiler_.record(SearchThreadRegion::kVirtualBackprop);

  if (mcts::kEnableDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str();
  }

  for (int i = search_path_.size() - 1; i >= 0; --i) {
    Node* node = search_path_[i].node;
    node->update_stats(VirtualIncrement{});
  }
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::pure_backprop(const ValueArray& value) {
  profiler_.record(SearchThreadRegion::kPureBackprop);

  if (mcts::kEnableDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str() << " "
             << value.transpose();
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

template <core::concepts::Game Game>
void SearchThread<Game>::backprop_with_virtual_undo(const ValueArray& value) {
  profiler_.record(SearchThreadRegion::kBackpropWithVirtualUndo);

  if (mcts::kEnableDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str() << " "
             << value.transpose();
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

template <core::concepts::Game Game>
void SearchThread<Game>::short_circuit_backprop(edge_t* last_edge) {
  // short-circuit
  if (mcts::kEnableDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str();
  }

  last_edge->increment_count();

  for (int i = search_path_.size() - 1; i >= 0; --i) {
    Node* child = search_path_[i].node;
    edge_t* edge = search_path_[i].edge;
    child->update_stats(RealIncrement{});
    if (i) edge->increment_count();
  }
}

template <core::concepts::Game Game>
typename SearchThread<Game>::evaluation_result_t
SearchThread<Game>::evaluate(Node* node) {
  profiler_.record(SearchThreadRegion::kEvaluate);

  std::unique_lock<std::mutex> lock(node->evaluation_data_mutex());
  typename Node::evaluation_data_t& evaluation_data = node->evaluation_data();
  evaluation_result_t data{evaluation_data.ptr.load(), false};
  auto state = evaluation_data.state;

  switch (state) {
    case Node::kUnset: {
      evaluate_unset(node, &lock, &data);
      node->cv_evaluate().notify_all();
      break;
    }
    default:
      break;
  }
  return data;
}

template <core::concepts::Game Game>
void SearchThread<Game>::evaluate_unset(Node* node, std::unique_lock<std::mutex>* lock,
                                        evaluation_result_t* data) {
  profiler_.record(SearchThreadRegion::kEvaluateUnset);

  if (mcts::kEnableDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str();
  }

  data->backpropagated_virtual_loss = true;
  util::debug_assert(data->evaluation.get() == nullptr);

  auto& evaluation_data = node->evaluation_data();

  virtual_backprop();

  const auto& stable_data = node->stable_data();
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
    typename NNEvaluationService::Request request{node,       &state_,    &state_history_,
                                                  &profiler_, thread_id_, sym_index};
    auto response = nn_eval_service_->evaluate(request);
    data->evaluation = response.ptr;
  }

  LocalPolicyArray P = eigen_util::softmax(data->evaluation->local_policy_logit_distr());
  if (node == shared_data_->root_node.get()) {
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

template <core::concepts::Game Game>
std::string SearchThread<Game>::search_path_str() const {
  std::string delim = IO::action_delimiter();
  std::vector<std::string> vec;
  for (int n = 1; n < (int)search_path_.size(); ++n) {  // skip the first node
    core::action_t action = search_path_[n].edge->action();
    vec.push_back(IO::action_to_str(action));
  }
  return util::create_string("[%s]", boost::algorithm::join(vec, delim).c_str());
}

template <core::concepts::Game Game>
core::action_index_t SearchThread<Game>::get_best_action_index(
    Node* node, NNEvaluation* evaluation) {
  profiler_.record(SearchThreadRegion::kPUCT);

  const SearchParams& search_params = shared_data_->search_params;
  PUCTStats stats(*manager_params_, search_params, node, node == shared_data_->root_node.get());

  using PVec = LocalPolicyArray;

  const PVec& P = stats.P;
  const PVec& N = stats.N;
  PVec& PUCT = stats.PUCT;

  bool add_noise = !search_params.disable_exploration && manager_params_->dirichlet_mult > 0;
  if (manager_params_->forced_playouts && add_noise) {
    PVec n_forced = (P * manager_params_->k_forced * N.sum()).sqrt();
    auto F1 = (N < n_forced).template cast<float>();
    auto F2 = (N > 0).template cast<float>();
    auto F = F1 * F2;
    PUCT = PUCT * (1 - F) + F * 1e+6;
  }

  int argmax_index;
  PUCT.maxCoeff(&argmax_index);

  if (mcts::kEnableDebug) {
    std::ostringstream ss;
    ss << thread_id_whitespace();

    const auto& tree_stats = node->stats();
    core::seat_index_t cp = node->stable_data().current_player;

    using ArrayT1 = Eigen::Array<float, 3, kNumPlayers>;
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

    ss << "real_avg: " << s1_lines[0] << break_plus_thread_id_whitespace();
    ss << "virt_avg: " << s1_lines[1] << break_plus_thread_id_whitespace();

    std::string cp_line = s1_lines[2];
    std::replace(cp_line.begin(), cp_line.end(), '0', ' ');
    std::replace(cp_line.begin(), cp_line.end(), '1', '*');

    ss << "cp:       " << cp_line << break_plus_thread_id_whitespace();

    using ScalarT = typename PVec::Scalar;
    constexpr int kNumRows = 10;  // action, P, V, PW, PL, E, N, VN, PUCT, argmax
    constexpr int kMaxCols = PVec::MaxRowsAtCompileTime;
    using ArrayT2 = Eigen::Array<ScalarT, kNumRows, Eigen::Dynamic, 0, kNumRows, kMaxCols>;

    ArrayT2 A2(kNumRows, P.rows());
    A2.setZero();

    const ActionMask& valid_actions = node->stable_data().valid_action_mask;
    int r = 0;
    int c = 0;
    for (int i : bitset_util::on_indices(valid_actions)) {
      A2(r, c++) = i;
    }
    r++;

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
    ss << "move:  " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "P:     " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "V:     " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "PW:    " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "PL:    " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "E:     " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "N:     " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "VN:    " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "PUCT:  " << s2_lines[r++] << break_plus_thread_id_whitespace();

    std::string argmax_line = s2_lines[r];
    std::replace(argmax_line.begin(), argmax_line.end(), '0', ' ');
    std::replace(argmax_line.begin(), argmax_line.end(), '1', '*');

    ss << "argmax:" << argmax_line << break_plus_thread_id_whitespace();
    ss << "*************";

    LOG_INFO << ss.str();
  }
  return argmax_index;
}

}  // namespace mcts
