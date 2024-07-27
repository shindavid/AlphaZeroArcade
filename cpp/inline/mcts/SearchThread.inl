#include <mcts/SearchThread.hpp>

#include <mcts/NNEvaluationRequest.hpp>
#include <util/Asserts.hpp>
#include <util/CppUtil.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

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
inline void SearchThread<Game>::state_data_t::load(const FullState& s, const base_state_vec_t& h) {
  state = s;
  state_history = h;
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::state_data_t::add_state_to_history() {
  util::stuff_back<Game::Constants::kHistorySize>(state_history, state);
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::state_data_t::canonical_validate() {
  constexpr bool check = IS_MACRO_ENABLED(DEBUG_BUILD);

  if (!check) return;
  group::element_t e = Game::Symmetries::get_canonical_symmetry(state);
  if (e == group::kIdentity) return;

  std::cout << "canonical_validate() failure (e=" << e << ")" << std::endl;
  Game::IO::print_state(std::cout, state);
  throw std::runtime_error("canonical_validate() failure");
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

  node_pool_index_t root_index = shared_data_->root_info.node_index;
  Node* root = shared_data_->lookup_table.get_node(root_index);
  if (root->is_terminal() || root->edges_initialized()) return root;

  const FullState& state = shared_data_->root_info.state[canonical_sym_];
  const base_state_vec_t& state_history = shared_data_->root_info.state_history[canonical_sym_];

  root->initialize_edges(state);
  canonical_state_data_.load(state, state_history);
  init_node(&canonical_state_data_, root_index, root);

  root->stats().RN++;
  return root;
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::init_node(state_data_t* state_data, node_pool_index_t index,
                                          Node* node) {
  FullState* state = &state_data->state;
  base_state_vec_t* state_history = &state_data->state_history;
  state_data->canonical_validate();

  if (!node->is_terminal()) {
    NNEvaluation* eval = nullptr;
    if (nn_eval_service_) {
      group::element_t sym = 0;
      if (manager_params_->apply_random_symmetries) {
        auto mask = Game::Symmetries::get_mask(*state);
        sym = bitset_util::choose_random_on_index(mask);
      }
      NNEvaluationRequest request(node, state, state_history, &profiler_, thread_id_, sym);
      eval = nn_eval_service_->evaluate(request).get();
    }
    node->load_eval(eval, [&](LocalPolicyArray& P) { transform_policy(index, P); });
  }

  auto mcts_key = Game::InputTensorizor::mcts_key(*state);
  shared_data_->lookup_table.insert_node(mcts_key, index);
}

template <core::concepts::Game Game>
void SearchThread<Game>::transform_policy(node_pool_index_t index, LocalPolicyArray& P) const {
  if (index == shared_data_->root_info.node_index) {
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
  const auto& root_info = shared_data_->root_info;
  canonical_sym_ = root_info.canonical_sym;
  constexpr group::element_t e = group::kIdentity;
  raw_state_data_.load(root_info.state[e], root_info.state_history[e]);

  Node* root = init_root_node();
  while (root->stats().total_count() <= shared_data_->search_params.tree_size_limit) {
    search_path_.clear();
    visit(root, nullptr);
    canonical_sym_ = root_info.canonical_sym;
    raw_state_data_.load(root_info.state[e], root_info.state_history[e]);
    dump_profiling_stats();
    if (!shared_data_->search_params.ponder && root->trivial()) break;
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
    ss << thread_id_whitespace() << "visit " << search_path_str()
       << " cp=" << (int)node->stable_data().current_player;
    LOG_INFO << ss.str();
  }
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::visit(Node* node, edge_t* parent_edge) {
  using Group = Game::SymmetryGroup;
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
  group::element_t inv_canonical_sym = Group::inverse(canonical_sym_);
  if (edge->state != Node::kExpanded) {
    // reread state under mutex in case of race-condition
    std::unique_lock lock(node->mutex());

    if (edge->state == Node::kNotExpanded) {
      edge->state = Node::kMidExpansion;
      lock.unlock();

      // reorient edge->action into raw-orientation
      core::action_t edge_action = edge->action;
      Game::Symmetries::apply(edge_action, inv_canonical_sym);

      // apply raw-orientation action to raw-orientation leaf-state
      outcome_ = Game::Rules::apply(raw_state_data_.state, edge_action);
      raw_state_data_.add_state_to_history();

      // determine canonical orientation of new leaf-state
      group::element_t new_sym = Game::Symmetries::get_canonical_symmetry(raw_state_data_.state);
      edge->sym = Group::compose(new_sym, inv_canonical_sym);

      canonical_sym_ = new_sym;
      applied_action = true;

      state_data_t* state_data = &raw_state_data_;
      if (canonical_sym_ != group::kIdentity) {
        calc_canonical_state_data();
        state_data = &canonical_state_data_;
      }

      if (expand(state_data, node, edge)) return;
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
    // reorient edge->action into raw-orientation
    core::action_t edge_action = edge->action;
    Game::Symmetries::apply(edge_action, inv_canonical_sym);

    outcome_ = Game::Rules::apply(raw_state_data_.state, edge_action);
    raw_state_data_.add_state_to_history();
    canonical_sym_ = Group::compose(edge->sym, canonical_sym_);
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
bool SearchThread<Game>::expand(state_data_t* state_data, Node* parent, edge_t* edge) {
  using LookupTable = Node::LookupTable;
  using MCTSKey = Game::InputTensorizor::MCTSKey;

  profiler_.record(SearchThreadRegion::kExpand);

  state_data->canonical_validate();
  const FullState& state = state_data->state;

  LookupTable& lookup_table = shared_data_->lookup_table;
  MCTSKey mcts_key = Game::InputTensorizor::mcts_key(state);
  node_pool_index_t child_index = lookup_table.lookup_node(mcts_key);

  bool is_new_node = child_index < 0;
  if (is_new_node) {
    edge->child_index = lookup_table.alloc_node();
    Node* child = lookup_table.get_node(edge->child_index);
    new (child) Node(&lookup_table, state, outcome_);
    search_path_.back().child = child;
    child->initialize_edges(state);
    virtual_backprop();
    init_node(state_data, edge->child_index, child);
    backprop_with_virtual_undo();
  } else {
    edge->child_index = child_index;
  }

  std::unique_lock lock(parent->mutex());
  parent->update_child_expand_count();
  edge->state = Node::kExpanded;
  lock.unlock();

  parent->cv().notify_all();
  return is_new_node;
}

template <core::concepts::Game Game>
std::string SearchThread<Game>::search_path_str() const {
  group::element_t cur_sym = shared_data_->root_info.canonical_sym;
  std::string delim = Game::IO::action_delimiter();
  std::vector<std::string> vec;
  for (const visitation_t& visitation : search_path_) {
    core::action_t action = visitation.edge->action;
    Game::Symmetries::apply(action, cur_sym);
    cur_sym = Game::SymmetryGroup::compose(visitation.edge->sym, cur_sym);
    vec.push_back(Game::IO::action_to_str(action));
  }
  return util::create_string("[%s]", boost::algorithm::join(vec, delim).c_str());
}

template <core::concepts::Game Game>
void SearchThread<Game>::calc_canonical_state_data() {
  canonical_state_data_ = raw_state_data_;
  for (BaseState& base : canonical_state_data_.state_history) {
    Game::Symmetries::apply(base, canonical_sym_);
  }

  if constexpr (core::concepts::RequiresMctsDoublePass<Game>) {
    using Group = Game::SymmetryGroup;
    canonical_state_data_.state = shared_data_->root_info.state[canonical_sym_];
    group::element_t cur_canonical_sym = shared_data_->root_info.canonical_sym;
    group::element_t leaf_canonical_sym = canonical_sym_;
    for (const visitation_t& visitation : search_path_) {
      edge_t* edge = visitation.edge;
      core::action_t action = edge->action;
      group::element_t sym = Group::compose(leaf_canonical_sym, Group::inverse(cur_canonical_sym));
      Game::Symmetries::apply(action, sym);
      Game::Rules::apply(canonical_state_data_.state, action);
      cur_canonical_sym = Group::compose(edge->sym, cur_canonical_sym);
    }

    util::release_assert(cur_canonical_sym == leaf_canonical_sym,
                         "cur_canonical_sym=%d leaf_canonical_sym=%d", cur_canonical_sym,
                         leaf_canonical_sym);
  } else {
    Game::Symmetries::apply(canonical_state_data_.state, canonical_sym_);
  }
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

  print_puct_details(node, stats, argmax_index);
  return argmax_index;
}

template <core::concepts::Game Game>
void SearchThread<Game>::print_puct_details(Node* node, const PUCTStats& stats,
                                            int argmax_index) const {
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

    using PVec = LocalPolicyArray;
    using ScalarT = PVec::Scalar;
    constexpr int kNumRows = 12;  // action, P, V, FPU, PW, PL, E, N, VN, &ch, PUCT, argmax
    constexpr int kMaxCols = PVec::MaxRowsAtCompileTime;
    using ArrayT2 = Eigen::Array<ScalarT, kNumRows, Eigen::Dynamic, 0, kNumRows, kMaxCols>;

    ArrayT2 A2(kNumRows, stats.P.rows());
    A2.setZero();

    int r = 0;

    PVec child_addr(stats.P.rows());
    child_addr.setConstant(-1);

    group::element_t inv_sym = Game::SymmetryGroup::inverse(canonical_sym_);
    for (int e = 0; e < node->stable_data().num_valid_actions; ++e) {
      auto edge = node->get_edge(e);
      core::action_t action = edge->action;
      Game::Symmetries::apply(action, inv_sym);
      A2(r, e) = action;
      child_addr(e) = edge->child_index;
    }
    r++;

    A2.row(r++) = stats.P;
    A2.row(r++) = stats.V;
    A2.row(r++) = stats.FPU;
    A2.row(r++) = stats.PW;
    A2.row(r++) = stats.PL;
    A2.row(r++) = stats.E;
    A2.row(r++) = stats.N;
    A2.row(r++) = stats.VN;
    A2.row(r++) = child_addr;
    A2.row(r++) = stats.PUCT;
    A2(r, argmax_index) = 1;

    A2 = eigen_util::sort_columns(A2);

    std::ostringstream ss2;
    ss2 << A2;
    std::string s2 = ss2.str();

    std::vector<std::string> s2_lines;
    boost::split(s2_lines, s2, boost::is_any_of("\n"));

    r = 0;
    ss << "move:  " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "P:     " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "V:     " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "FPU:   " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "PW:    " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "PL:    " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "E:     " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "N:     " << s2_lines[r++] << break_plus_thread_id_whitespace();
    ss << "VN:    " << s2_lines[r++] << break_plus_thread_id_whitespace();

    std::string ch_line = s2_lines[r++];
    boost::replace_all(ch_line, "-1", "  ");

    ss << "&ch:   " << ch_line << break_plus_thread_id_whitespace();
    ss << "PUCT:  " << s2_lines[r++] << break_plus_thread_id_whitespace();

    std::string argmax_line = s2_lines[r];
    std::replace(argmax_line.begin(), argmax_line.end(), '0', ' ');
    std::replace(argmax_line.begin(), argmax_line.end(), '1', '*');

    ss << "argmax:" << argmax_line << break_plus_thread_id_whitespace();
    ss << "*************";

    LOG_INFO << ss.str();
  }
}

}  // namespace mcts
