#include <mcts/SearchThread.hpp>

#include <mcts/Node.hpp>
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
                                        NNEvaluationServiceBase* nn_eval_service,
                                        const ManagerParams* manager_params,
                                        int thread_id)
    : shared_data_(shared_data),
      nn_eval_service_(nn_eval_service),
      manager_params_(manager_params),
      thread_id_(thread_id) {}

template <core::concepts::Game Game>
inline SearchThread<Game>::~SearchThread() {
  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
  profiler_.dump(1);
  profiler_.close_file();
}

template <core::concepts::Game Game>
void SearchThread<Game>::start() {
  util::release_assert(thread_ == nullptr);
  thread_ = new std::thread([=, this] { loop(); });
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

  node_pool_index_t root_index = shared_data_->root_info.node_index;
  Node* root = shared_data_->lookup_table.get_node(root_index);
  if (root->is_terminal()) return root;

  if (!root->edges_initialized()) {
    root->initialize_edges();
  }

  if (root->all_children_edges_initialized()) {
    return root;
  }

  StateHistory& history = shared_data_->root_info.history_array[canonical_sym_];

  pseudo_local_vars_.canonical_history = history;
  init_node(&pseudo_local_vars_.canonical_history, root_index, root);

  return root;
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::init_node(StateHistory* history, node_pool_index_t index,
                                          Node* node) {
  if (!node->is_terminal()) {
    bool is_root = (node == shared_data_->get_root_node());
    bool eval_all_children = manager_params_->force_evaluate_all_root_children && is_root &&
                             shared_data_->search_params.full_search;

    const State& state = history->current();
    NNEvaluationRequest request(pseudo_local_vars_.request_items, &profiler_, thread_id_);

    if (!node->stable_data().VT_valid) {
      group::element_t sym = group::kIdentity;
      if (manager_params_->apply_random_symmetries) {
        sym = bitset_util::choose_random_on_index(Game::Symmetries::get_mask(state));
      }
      bool incorporate = manager_params_->incorporate_sym_into_cache_key;
      pseudo_local_vars_.request_items.emplace_back(node, *history, sym, incorporate);
    }
    if (eval_all_children) {
      expand_all_children(node, &request);
    }

    if (!pseudo_local_vars_.request_items.empty()) {
      nn_eval_service_->evaluate(request);

      for (auto& item : pseudo_local_vars_.request_items) {
        item.node()->load_eval(item.eval(),
                               [&](LocalPolicyArray& P) { transform_policy(index, P); });
      }
      pseudo_local_vars_.request_items.clear();
    }

    if (node->stable_data().is_chance_node) {
      for (int i = 0; i < node->stable_data().num_valid_actions; i++) {
        edge_t* edge = node->get_edge(i);
        core::action_t action = edge->action;
        typename Game::Types::ChanceDistribution chance_dist =
            Game::Rules::get_chance_distribution(history->current());
        edge->chance_prob = chance_dist(action);
      }
    }
  }

  auto mcts_key = Game::InputTensorizor::mcts_key(*history);
  shared_data_->lookup_table.insert_node(mcts_key, index);
}

template <core::concepts::Game Game>
void SearchThread<Game>::expand_all_children(Node* node, NNEvaluationRequest* request) {
  using Group = Game::SymmetryGroup;

  LookupTable& lookup_table = shared_data_->lookup_table;
  group::element_t inv_canonical_sym = Group::inverse(canonical_sym_);

  // Evaluate every child of the root node
  int n_actions = node->stable_data().num_valid_actions;
  int expand_count = 0;
  for (int e = 0; e < n_actions; e++) {
    edge_t* edge = node->get_edge(e);
    if (edge->child_index >= 0) continue;

    // reorient edge->action into raw-orientation
    core::action_t raw_edge_action = edge->action;
    Game::Symmetries::apply(raw_edge_action, inv_canonical_sym, node->action_mode());

    // apply raw-orientation action to raw-orientation child-state
    Game::Rules::apply(raw_history_, raw_edge_action);

    // determine canonical orientation of new leaf-state
    group::element_t canonical_child_sym =
        Game::Symmetries::get_canonical_symmetry(raw_history_.current());
    edge->sym = Group::compose(canonical_child_sym, inv_canonical_sym);

    StateHistory& canonical_history = pseudo_local_vars_.root_history_array[canonical_child_sym];

    core::action_t reoriented_action = raw_edge_action;
    Game::Symmetries::apply(reoriented_action, canonical_child_sym, node->action_mode());
    Game::Rules::apply(canonical_history, reoriented_action);

    expand_count++;
    edge->state = Node::kPreExpanded;

    MCTSKey mcts_key = Game::InputTensorizor::mcts_key(canonical_history);
    node_pool_index_t child_index = lookup_table.lookup_node(mcts_key);
    if (child_index >= 0) {
      edge->child_index = child_index;
      canonical_history.undo();
      raw_history_.undo();
      continue;
    }

    edge->child_index = lookup_table.alloc_node();
    Node* child = lookup_table.get_node(edge->child_index);

    ValueTensor game_outcome;
    if (Game::Rules::is_terminal(raw_history_.current(), node->stable_data().current_player,
                                 raw_edge_action, game_outcome)) {
      new (child) Node(&lookup_table, canonical_history, game_outcome);
    } else {
      new (child) Node(&lookup_table, canonical_history);
    }
    child->initialize_edges();
    shared_data_->lookup_table.insert_node(mcts_key, edge->child_index);

    State canonical_child_state = canonical_history.current();
    canonical_history.undo();
    raw_history_.undo();

    if (child->is_terminal()) continue;
    if (!request) continue;

    group::element_t sym = group::kIdentity;
    if (manager_params_->apply_random_symmetries) {
      sym = bitset_util::choose_random_on_index(Game::Symmetries::get_mask(canonical_child_state));
    }
    bool incorporate = manager_params_->incorporate_sym_into_cache_key;
    pseudo_local_vars_.request_items.emplace_back(child, canonical_history, canonical_child_state,
                                                  sym, incorporate);
  }

  node->update_child_expand_count(expand_count);
}

template <core::concepts::Game Game>
void SearchThread<Game>::transform_policy(node_pool_index_t index, LocalPolicyArray& P) const {
  if (index == shared_data_->root_info.node_index) {
    if (shared_data_->search_params.full_search) {
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
  pseudo_local_vars_.root_history_array = root_info.history_array;
  canonical_sym_ = root_info.canonical_sym;
  raw_history_ = root_info.history_array[group::kIdentity];

  Node* root = init_root_node();
  while (root->stats().total_count() <= shared_data_->search_params.tree_size_limit) {
    search_path_.clear();
    search_path_.emplace_back(root, nullptr);
    visit(root);
    root->validate_state();
    canonical_sym_ = root_info.canonical_sym;
    raw_history_ = root_info.history_array[group::kIdentity];
    dump_profiling_stats();
    if (!shared_data_->search_params.ponder && root->trivial()) break;
    post_visit_func();
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
void SearchThread<Game>::print_visit_info(Node* node) {
  if (mcts::kEnableSearchDebug) {
    std::ostringstream ss;
    ss << thread_id_whitespace() << "visit " << search_path_str()
       << " cp=" << (int)node->stable_data().current_player;
    LOG_INFO << ss.str();
  }
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::visit(Node* node) {
  using Group = Game::SymmetryGroup;
  print_visit_info(node);

  const auto& stable_data = node->stable_data();
  if (stable_data.terminal) {
    pure_backprop(Game::GameResults::to_value_array(stable_data.VT));
    return;
  }

  int child_index;
  if (stable_data.is_chance_node) {
    child_index = sample_chance_child_index(node);
  } else {
    child_index = get_best_child_index(node);
  }

  edge_t* edge = node->get_edge(child_index);
  search_path_.back().edge = edge;
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
      Game::Symmetries::apply(edge_action, inv_canonical_sym, node->action_mode());

      // apply raw-orientation action to raw-orientation leaf-state
      Game::Rules::apply(raw_history_, edge_action);

      // determine canonical orientation of new leaf-state
      group::element_t new_sym = Game::Symmetries::get_canonical_symmetry(raw_history_.current());
      edge->sym = Group::compose(new_sym, inv_canonical_sym);

      canonical_sym_ = new_sym;
      applied_action = true;

      StateHistory* state_history = &raw_history_;
      if (canonical_sym_ != group::kIdentity) {
        calc_canonical_state_data();
        state_history = &pseudo_local_vars_.canonical_history;
      }

      if (expand(state_history, node, edge)) return;
    } else if (edge->state == Node::kMidExpansion) {
      node->cv().wait(lock, [edge] { return edge->state == Node::kExpanded; });
    } else if (edge->state == Node::kPreExpanded) {
      edge->state = Node::kMidExpansion;
      lock.unlock();

      util::debug_assert(edge->child_index >= 0);
      Node* child = shared_data_->lookup_table.get_node(edge->child_index);
      search_path_.emplace_back(child, nullptr);
      int edge_count = edge->E;
      int child_count = child->stats().RN;
      if (edge_count < child_count) {
        short_circuit_backprop();
      } else {
        standard_backprop(false);
      }

      lock.lock();
      edge->state = Node::kExpanded;
      lock.unlock();
      node->cv().notify_all();
      return;
    }
  }

  util::release_assert(edge->state == Node::kExpanded);
  Node* child = node->get_child(edge);
  if (child) {
    search_path_.emplace_back(child, nullptr);
    int edge_count = edge->E;
    int child_count = child->stats().RN;
    if (edge_count < child_count) {
      short_circuit_backprop();
      return;
    }
  }
  if (!applied_action) {
    // reorient edge->action into raw-orientation
    core::action_t edge_action = edge->action;
    Game::Symmetries::apply(edge_action, inv_canonical_sym, node->action_mode());

    Game::Rules::apply(raw_history_, edge_action);
    canonical_sym_ = Group::compose(edge->sym, canonical_sym_);
  }
  visit(child);
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::add_dirichlet_noise(LocalPolicyArray& P) const {
  int n = P.rows();
  double alpha = manager_params_->dirichlet_alpha_factor / sqrt(n);
  LocalPolicyArray noise = dirichlet_gen().template generate<LocalPolicyArray>(rng(), alpha, n);
  P = (1.0 - manager_params_->dirichlet_mult) * P + manager_params_->dirichlet_mult * noise;
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::virtual_backprop() {
  profiler_.record(SearchThreadRegion::kVirtualBackprop);

  if (mcts::kEnableSearchDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str();
  }

  util::release_assert(!search_path_.empty());
  Node* last_node = search_path_.back().node;

  last_node->update_stats([&] {
    last_node->stats().VN++;
  });

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    edge_t* edge = search_path_[i].edge;
    Node* node = search_path_[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().VN++;
    });
  }
  validate_search_path();
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::pure_backprop(const ValueArray& value) {
  profiler_.record(SearchThreadRegion::kPureBackprop);

  if (mcts::kEnableSearchDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str() << " "
             << value.transpose();
  }

  util::release_assert(!search_path_.empty());
  Node* last_node = search_path_.back().node;

  last_node->update_stats([&] {
    last_node->stats().init_q(value, true);
    last_node->stats().RN++;
  });

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    edge_t* edge = search_path_[i].edge;
    Node* node = search_path_[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().RN++;
    });
  }
  validate_search_path();
}

template <core::concepts::Game Game>
void SearchThread<Game>::standard_backprop(bool undo_virtual) {
  profiler_.record(SearchThreadRegion::kBackpropWithVirtualUndo);

  Node* last_node = search_path_.back().node;
  auto value = Game::GameResults::to_value_array(last_node->stable_data().VT);

  if (mcts::kEnableSearchDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str() << ": "
             << value.transpose();
  }

  last_node->update_stats([&] {
    last_node->stats().init_q(value, false);
    last_node->stats().RN++;
    last_node->stats().VN -= undo_virtual;
  });

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    edge_t* edge = search_path_[i].edge;
    Node* node = search_path_[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E += !undo_virtual;
      node->stats().RN++;
      node->stats().VN -= undo_virtual;
    });
  }
  validate_search_path();
}

template <core::concepts::Game Game>
void SearchThread<Game>::short_circuit_backprop() {
  if (mcts::kEnableSearchDebug) {
    LOG_INFO << thread_id_whitespace() << __func__ << " " << search_path_str();
  }

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    edge_t* edge = search_path_[i].edge;
    Node* node = search_path_[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().RN++;
    });
  }
  validate_search_path();
}

template <core::concepts::Game Game>
bool SearchThread<Game>::expand(StateHistory* history, Node* parent, edge_t* edge) {
  profiler_.record(SearchThreadRegion::kExpand);

  LookupTable& lookup_table = shared_data_->lookup_table;
  MCTSKey mcts_key = Game::InputTensorizor::mcts_key(*history);
  node_pool_index_t child_index = lookup_table.lookup_node(mcts_key);

  bool is_new_node = child_index < 0;
  if (is_new_node) {
    edge->child_index = lookup_table.alloc_node();
    Node* child = lookup_table.get_node(edge->child_index);

    ValueTensor game_outcome;
    core::action_t last_action = edge->action;
    Game::Symmetries::apply(last_action, edge->sym, parent->action_mode());
    core::seat_index_t last_player = 1 - Game::Rules::get_current_player(history->current());

    bool terminal = Game::Rules::is_terminal(
        history->current(), last_player, last_action, game_outcome);

    if (terminal) {
      new (child) Node(&lookup_table, *history, game_outcome);
    } else {
      new (child) Node(&lookup_table, *history);
    }

    search_path_.emplace_back(child, nullptr);
    child->initialize_edges();
    bool do_virtual = !terminal && manager_params_->num_search_threads > 1;
    if (do_virtual) {
      virtual_backprop();
    }
    init_node(history, edge->child_index, child);
    if (terminal) {
      pure_backprop(Game::GameResults::to_value_array(child->stable_data().VT));
    } else {
      standard_backprop(do_virtual);
    }
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
  using Group = Game::SymmetryGroup;
  group::element_t cur_sym = Group::inverse(shared_data_->root_info.canonical_sym);
  std::string delim = Game::IO::action_delimiter();
  std::vector<std::string> vec;
  for (const visitation_t& visitation : search_path_) {
    if (!visitation.edge) continue;
    core::action_mode_t mode = visitation.node->action_mode();
    core::action_t action = visitation.edge->action;
    Game::Symmetries::apply(action, cur_sym, mode);
    cur_sym = Group::compose(cur_sym, Group::inverse(visitation.edge->sym));
    vec.push_back(Game::IO::action_to_str(action, mode));
  }
  return util::create_string("[%s]", boost::algorithm::join(vec, delim).c_str());
}

template <core::concepts::Game Game>
void SearchThread<Game>::calc_canonical_state_data() {
  pseudo_local_vars_.canonical_history = raw_history_;

  if constexpr (core::concepts::RequiresMctsDoublePass<Game>) {
    using Group = Game::SymmetryGroup;
    pseudo_local_vars_.canonical_history = shared_data_->root_info.history_array[canonical_sym_];
    group::element_t cur_canonical_sym = shared_data_->root_info.canonical_sym;
    group::element_t leaf_canonical_sym = canonical_sym_;
    for (const visitation_t& visitation : search_path_) {
      edge_t* edge = visitation.edge;
      core::action_mode_t mode = visitation.node->action_mode();
      core::action_t action = edge->action;
      group::element_t sym = Group::compose(leaf_canonical_sym, Group::inverse(cur_canonical_sym));
      Game::Symmetries::apply(action, sym, mode);
      Game::Rules::apply(pseudo_local_vars_.canonical_history, action);
      cur_canonical_sym = Group::compose(edge->sym, cur_canonical_sym);
    }

    util::release_assert(cur_canonical_sym == leaf_canonical_sym,
                         "cur_canonical_sym=%d leaf_canonical_sym=%d", cur_canonical_sym,
                         leaf_canonical_sym);
  } else {
    Game::Symmetries::apply(pseudo_local_vars_.canonical_history, canonical_sym_);
  }

  if (IS_MACRO_ENABLED(DEBUG_BUILD)) {
    State s = pseudo_local_vars_.canonical_history.current();
    Game::Symmetries::apply(s, Game::Symmetries::get_canonical_symmetry(s));
    if (s != pseudo_local_vars_.canonical_history.current()) {
      std::cout << "ERROR! Bad Canonicalization!" << std::endl;
      std::cout << "canonical_sym_: " << int(canonical_sym_) << std::endl;
      std::cout << "canonical_history.current():" << std::endl;
      Game::IO::print_state(std::cout, pseudo_local_vars_.canonical_history.current());
      std::cout << "Should be:" << std::endl;
      Game::IO::print_state(std::cout, s);
      util::release_assert(false);
    }
  }
}

template <core::concepts::Game Game>
void SearchThread<Game>::validate_search_path() const {
  if (!IS_MACRO_ENABLED(DEBUG_BUILD)) return;

  int N = search_path_.size();
  for (int i = N - 1; i >= 0; --i) {
    search_path_[i].node->validate_state();
  }
}

template <core::concepts::Game Game>
int SearchThread<Game>::get_best_child_index(Node* node) {
  profiler_.record(SearchThreadRegion::kPUCT);

  bool is_root = (node == shared_data_->get_root_node());
  const SearchParams& search_params = shared_data_->search_params;
  ActionSelector action_selector(*manager_params_, search_params, node, is_root);

  using PVec = LocalPolicyArray;

  const PVec& P = action_selector.P;
  const PVec& mE = action_selector.mE;
  PVec& PUCT = action_selector.PUCT;

  int argmax_index;

  if (search_params.tree_size_limit == 1) {
    // net-only, use P
    P.maxCoeff(&argmax_index);
  } else {
    bool force_playouts = manager_params_->forced_playouts && is_root &&
                          search_params.full_search && manager_params_->dirichlet_mult > 0;

    if (force_playouts) {
      PVec n_forced = (P * manager_params_->k_forced * mE.sum()).sqrt();
      auto F1 = (mE < n_forced).template cast<float>();
      auto F2 = (mE > 0).template cast<float>();
      auto F = F1 * F2;
      PUCT = PUCT * (1 - F) + F * 1e+6;
    }

    PUCT.maxCoeff(&argmax_index);
  }

  print_action_selection_details(node, action_selector, argmax_index);
  return argmax_index;
}

template <core::concepts::Game Game>
int SearchThread<Game>::sample_chance_child_index(Node* node) {
  bool is_root = (node == shared_data_->get_root_node());
  const SearchParams& search_params = shared_data_->search_params;
  ActionSelector action_selector(*manager_params_, search_params, node, is_root);
  const auto* chance_dist = action_selector.chance_dist;

  int n = Game::Constants::kMaxBranchingFactor;
  core::action_t chance_action = util::Random::weighted_sample(chance_dist, chance_dist + n);
  return chance_action;
}

template <core::concepts::Game Game>
void SearchThread<Game>::print_action_selection_details(Node* node, const ActionSelector& selector,
                                                        int argmax_index) const {
  if (mcts::kEnableSearchDebug) {
    std::ostringstream ss;
    ss << thread_id_whitespace();

    core::seat_index_t cp = node->stable_data().current_player;

    int n_actions = node->stable_data().num_valid_actions;

    ValueArray players;
    const ValueArray& nQ = node->stats().Q;
    ValueArray CP;
    for (int p = 0; p < kNumPlayers; ++p) {
      players(p) = p;
      CP(p) = p == cp;
    }

    std::vector<std::string> player_columns = {"Seat", "Q", "CurP"};
    auto player_data = eigen_util::concatenate_columns(players, nQ, CP);

    eigen_util::PrintArrayFormatMap fmt_map1;
    fmt_map1["Seat"] = [&](float x) { return std::to_string(int(x)); };
    fmt_map1["CurP"] = [&](float x) { return std::string(x == cp ? "*" : ""); };

    std::stringstream ss1;
    eigen_util::print_array(ss1, player_data, player_columns, &fmt_map1);

    for (const std::string& line : util::splitlines(ss1.str())) {
      ss << line << break_plus_thread_id_whitespace();
    }

    const LocalPolicyArray& P = selector.P;
    const LocalPolicyArray& Q = selector.Q;
    const LocalPolicyArray& FPU = selector.FPU;
    const LocalPolicyArray& PW = selector.PW;
    const LocalPolicyArray& PL = selector.PL;
    const LocalPolicyArray& E = selector.E;
    const LocalPolicyArray& mE = selector.mE;
    const LocalPolicyArray& RN = selector.RN;
    const LocalPolicyArray& VN = selector.VN;
    const LocalPolicyArray& PUCT = selector.PUCT;

    LocalPolicyArray actions(n_actions);
    LocalPolicyArray child_addr(n_actions);
    LocalPolicyArray argmax(n_actions);
    child_addr.setConstant(-1);
    argmax.setZero();
    argmax(argmax_index) = 1;

    group::element_t inv_sym = Game::SymmetryGroup::inverse(canonical_sym_);
    for (int e = 0; e < n_actions; ++e) {
      auto edge = node->get_edge(e);
      core::action_t action = edge->action;
      Game::Symmetries::apply(action, inv_sym, node->action_mode());
      actions(e) = action;
      child_addr(e) = edge->child_index;
    }

    std::vector<std::string> action_columns = {"action", "P",  "Q",  "FPU", "PW",   "PL",    "E",
                                               "mE",     "RN", "VN", "&ch", "PUCT", "argmax"};
    auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
        actions, P, Q, FPU, PW, PL, E, mE, RN, VN, child_addr, PUCT, argmax));

    eigen_util::PrintArrayFormatMap fmt_map2;
    fmt_map2["action"] = [&](float x) { return Game::IO::action_to_str(x, node->action_mode()); };
    fmt_map2["&ch"] = [](float x) { return x < 0 ? std::string() : std::to_string((int)x); };
    fmt_map2["argmax"] = [](float x) { return std::string(x == 0 ? "" : "*"); };

    std::stringstream ss2;
    eigen_util::print_array(ss2, action_data, action_columns, &fmt_map2);

    for (const std::string& line : util::splitlines(ss2.str())) {
      ss << line << break_plus_thread_id_whitespace();
    }

    LOG_INFO << ss.str();
  }
}

}  // namespace mcts
