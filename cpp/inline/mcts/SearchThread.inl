#include <mcts/SearchThread.hpp>

#include <mcts/NNEvaluationRequest.hpp>
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
Node<Game>* SearchThread<Game>::init_root_node() {
  std::unique_lock lock(shared_data_->init_root_mutex);

  node_pool_index_t root_index = shared_data_->root_node_index;
  Node* root = shared_data_->lookup_table.get_node(root_index);
  if (root->is_terminal() || root->edges_initialized()) return root;

  root->initialize_edges();
  init_node(root_index, root);
  root->stats().RN++;
  return root;
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::init_node(node_pool_index_t index, Node* node) {
  if (!node->is_terminal()) {
    NNEvaluation* eval = nullptr;
    if (nn_eval_service_) {
      group::element_t sym = 0;
      if (manager_params_->apply_random_symmetries) {
        sym = util::Random::uniform_sample(0, Game::SymmetryGroup::kOrder);
      }
      NNEvaluationRequest request(node, &state_, &state_history_, &profiler_, thread_id_, sym);
      eval = nn_eval_service_->evaluate(request).get();
    }
    node->load_eval(eval, [&](LocalPolicyArray& P) { transform_policy(index, P); });
  }

  auto mcts_key = Game::InputTensorizor::mcts_key(state_);
  shared_data_->lookup_table.insert_node(mcts_key, index);
}

template <core::concepts::Game Game>
void SearchThread<Game>::transform_policy(node_pool_index_t index, LocalPolicyArray& P) const {
  if (index == shared_data_->root_node_index) {
    if (!shared_data_->search_params.disable_exploration) {
      if (manager_params_->dirichlet_mult) {
        add_dirichlet_noise(P);
      }
      P = P.pow(1.0 / root_softmax_temperature());
      P /= P.sum();
    }
  }
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::perform_visits() {
  state_ = shared_data_->root_state;
  state_history_ = shared_data_->root_state_history;
  Node* root = init_root_node();
  while (root->stats().total_count() <= shared_data_->search_params.tree_size_limit) {
    search_path_.clear();
    visit(root, nullptr);
    state_ = shared_data_->root_state;
    state_history_ = shared_data_->root_state_history;
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
void SearchThread<Game>::print_visit_info(Node* node, edge_t* parent_edge) {
  if (mcts::kEnableDebug) {
    std::ostringstream ss;
    ss << thread_id_whitespace();
    if (parent_edge) {
      ss << "visit " << parent_edge->action;
    } else {
      ss << "visit()";
    }
    ss << " " << search_path_str() << " cp=" << (int)node->stable_data().current_player;
    LOG_INFO << ss.str();
  }
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::visit(Node* node, edge_t* parent_edge) {
  print_visit_info(node, parent_edge);

  const auto& stable_data = node->stable_data();
  if (stable_data.terminal) {
    pure_backprop(stable_data.V);
    return;
  }

  int child_index = get_best_child_index(node);
  edge_t* edge = node->get_edge(child_index);
  search_path_.emplace_back(edge, nullptr);
  bool applied_action = false;
  if (edge->state != Node::kExpanded) {
    // reread state under mutex in case of race-condition
    std::unique_lock lock(node->mutex());

    if (edge->state == Node::kNotExpanded) {
      edge->state = Node::kMidExpansion;
      lock.unlock();

      outcome_ = Game::Rules::apply(state_, edge->action);
      util::stuff_back<Game::Constants::kHistorySize>(state_history_, state_);
      applied_action = true;
      if (expand(node, edge)) return;
    } else if (edge->state == Node::kMidExpansion) {
      node->cv().wait(lock, [edge] { return edge->state == Node::kExpanded; });
    }
  }

  util::release_assert(edge->state == Node::kExpanded);
  Node* child = node->get_child(edge);
  search_path_.back().child = child;
  if (child) {
    int edge_count = edge->RN;
    int child_count = child->stats().RN;
    if (edge_count < child_count) {
      short_circuit_backprop();
      return;
    }
  }
  if (!applied_action) {
    outcome_ = Game::Rules::apply(state_, edge->action);
    util::stuff_back<Game::Constants::kHistorySize>(state_history_, state_);
  }
  visit(child, edge);
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::add_dirichlet_noise(LocalPolicyArray& P) const {
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
    search_path_[i].child->update_stats(VirtualIncrement{});
  }
  shared_data_->get_root_node()->update_stats(VirtualIncrement{});
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::pure_backprop(const ValueArray& value) {
  profiler_.record(SearchThreadRegion::kPureBackprop);

  if (mcts::kEnableDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str() << " "
             << value.transpose();
  }

  util::release_assert(!search_path_.empty());
  edge_t* last_edge = search_path_.back().edge;
  Node* last_node = search_path_.back().child;
  last_node->update_stats(InitQAndRealIncrement(value));
  last_edge->RN++;

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    edge_t* edge = search_path_[i].edge;
    Node* child = search_path_[i].child;
    child->update_stats(RealIncrement{});
    edge->RN++;
  }
  shared_data_->get_root_node()->update_stats(RealIncrement{});
}

template <core::concepts::Game Game>
void SearchThread<Game>::backprop_with_virtual_undo() {
  profiler_.record(SearchThreadRegion::kBackpropWithVirtualUndo);

  if (mcts::kEnableDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str();
  }

  edge_t* last_edge = search_path_.back().edge;
  Node* last_node = search_path_.back().child;
  auto value = last_node->stable_data().V;
  last_node->update_stats(InitQAndIncrementTransfer(value));
  last_edge->RN++;

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    edge_t* edge = search_path_[i].edge;
    Node* child = search_path_[i].child;
    child->update_stats(IncrementTransfer{});
    edge->RN++;
  }
  shared_data_->get_root_node()->update_stats(IncrementTransfer{});
}

template <core::concepts::Game Game>
void SearchThread<Game>::short_circuit_backprop() {
  if (mcts::kEnableDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str();
  }

  edge_t* last_edge = search_path_.back().edge;
  last_edge->RN++;

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    edge_t* edge = search_path_[i].edge;
    Node* child = search_path_[i].child;
    child->update_stats(RealIncrement{});
    edge->RN++;
  }
  shared_data_->get_root_node()->update_stats(RealIncrement{});
}

template <core::concepts::Game Game>
bool SearchThread<Game>::expand(Node* parent, edge_t* edge) {
  using LookupTable = Node::LookupTable;
  using MCTSKey = Game::InputTensorizor::MCTSKey;

  profiler_.record(SearchThreadRegion::kExpand);

  LookupTable& lookup_table = shared_data_->lookup_table;
  MCTSKey mcts_key = Game::InputTensorizor::mcts_key(state_);
  node_pool_index_t child_index = lookup_table.lookup_node(mcts_key);

  bool is_new_node = child_index < 0;
  if (is_new_node) {
    edge->child_index = lookup_table.alloc_node();
    Node* child = lookup_table.get_node(edge->child_index);
    new (child) Node(&lookup_table, state_, outcome_);
    search_path_.back().child = child;
    child->initialize_edges();
    virtual_backprop();
    init_node(edge->child_index, child);
    backprop_with_virtual_undo();
  } else {
    edge->child_index = child_index;
  }

  std::unique_lock lock(parent->mutex());
  edge->state = Node::kExpanded;
  lock.unlock();

  parent->cv().notify_all();
  return is_new_node;
}

template <core::concepts::Game Game>
std::string SearchThread<Game>::search_path_str() const {
  std::string delim = IO::action_delimiter();
  std::vector<std::string> vec;
  for (const visitation_t& visitation : search_path_) {
    vec.push_back(IO::action_to_str(visitation.edge->action));
  }
  return util::create_string("[%s]", boost::algorithm::join(vec, delim).c_str());
}

template <core::concepts::Game Game>
int SearchThread<Game>::get_best_child_index(Node* node) {
  profiler_.record(SearchThreadRegion::kPUCT);

  const SearchParams& search_params = shared_data_->search_params;
  PUCTStats stats(*manager_params_, search_params, node, node == shared_data_->get_root_node());

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

    core::seat_index_t cp = node->stable_data().current_player;

    using ArrayT1 = Eigen::Array<float, 3, kNumPlayers>;
    ArrayT1 A1;
    A1.setZero();
    A1.row(0) = node->stats().RQ;
    A1.row(1) = node->stats().VQ;
    A1(2, cp) = 1;

    std::ostringstream ss1;
    ss1 << A1;
    std::string s1 = ss1.str();

    std::vector<std::string> s1_lines;
    boost::split(s1_lines, s1, boost::is_any_of("\n"));

    ss << "RQ:    " << s1_lines[0] << break_plus_thread_id_whitespace();
    ss << "VQ:    " << s1_lines[1] << break_plus_thread_id_whitespace();

    std::string cp_line = s1_lines[2];
    std::replace(cp_line.begin(), cp_line.end(), '0', ' ');
    std::replace(cp_line.begin(), cp_line.end(), '1', '*');

    ss << "cp:    " << cp_line << break_plus_thread_id_whitespace();

    using ScalarT = PVec::Scalar;
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
