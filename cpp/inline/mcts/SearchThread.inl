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
      thread_id_(thread_id),
      multithreaded_(manager_params->num_search_threads > 1) {
  thread_id_whitespace_ = util::make_whitespace(kThreadWhitespaceLength * thread_id_);
  break_plus_thread_id_whitespace_ = util::create_string("\n%s", util::make_whitespace(
    util::Logging::kTimestampPrefixLength + kThreadWhitespaceLength * thread_id_).c_str());
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
typename SearchThread<Game>::node_pool_index_t
SearchThread<Game>::init_node(StateHistory* history, node_pool_index_t index, Node* node) {
  bool is_root = (node == shared_data_->get_root_node());
  if (!node->is_terminal()) {
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
      using ChanceDistribution = Game::Types::ChanceDistribution;
      ChanceDistribution chance_dist = Game::Rules::get_chance_distribution(history->current());
      for (int i = 0; i < node->stable_data().num_valid_actions; i++) {
        Edge* edge = node->get_edge(i);
        core::action_t action = edge->action;
        edge->base_prob = chance_dist(action);
      }
    }
  }

  auto mcts_key = Game::InputTensorizor::mcts_key(*history);
  bool overwrite = is_root;
  return shared_data_->lookup_table.insert_node(mcts_key, index, overwrite);
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
    Edge* edge = node->get_edge(e);
    if (edge->child_index >= 0) continue;

    // reorient edge->action into raw-orientation
    core::action_t raw_edge_action = edge->action;
    Game::Symmetries::apply(raw_edge_action, inv_canonical_sym, node->action_mode());

    // apply raw-orientation action to raw-orientation child-state
    Game::Rules::apply(raw_history_, raw_edge_action);

    const State& raw_child_state = raw_history_.current();

    // compute active-seat as local-variable, so we don't need an undo later
    core::action_mode_t child_mode = Game::Rules::get_action_mode(raw_child_state);
    core::seat_index_t child_active_seat = active_seat_;
    if (!Game::Rules::is_chance_mode(child_mode)) {
      child_active_seat = Game::Rules::get_current_player(raw_child_state);
    }

    // determine canonical orientation of new leaf-state
    group::element_t canonical_child_sym =
        Game::Symmetries::get_canonical_symmetry(raw_child_state);
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

    core::seat_index_t parent_active_seat = node->stable_data().active_seat;
    util::debug_assert(parent_active_seat == active_seat_);

    ValueTensor game_outcome;
    if (Game::Rules::is_terminal(raw_child_state, parent_active_seat,
                                 raw_edge_action, game_outcome)) {
      new (child) Node(&lookup_table, canonical_history, game_outcome);
    } else {
      new (child) Node(&lookup_table, canonical_history, child_active_seat);
    }
    child->initialize_edges();
    bool overwrite = false;
    shared_data_->lookup_table.insert_node(mcts_key, edge->child_index, overwrite);

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
  active_seat_ = root_info.active_seat;

  Node* root = init_root_node();

  // root->stats() usage here is not thread-safe but this race-condition is benign
  while (root->stats().total_count() <= shared_data_->search_params.tree_size_limit) {
    search_path_.clear();
    search_path_.emplace_back(root, nullptr);
    visit(root);
    root->validate_state();
    canonical_sym_ = root_info.canonical_sym;
    raw_history_ = root_info.history_array[group::kIdentity];
    active_seat_ = root_info.active_seat;
    dump_profiling_stats();
    if (!shared_data_->search_params.ponder && root->trivial()) break;
    post_visit_func();
  }
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::deactivate() const {
  std::unique_lock lock(shared_data_->search_mutex);
  shared_data_->active_search_threads[thread_id_] = 0;
  lock.unlock();
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
    LOG_INFO("{}visit {} seat={}", thread_id_whitespace(), search_path_str(),
             node->stable_data().active_seat);
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

  Edge* edge = node->get_edge(child_index);
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

      core::action_mode_t child_mode = Game::Rules::get_action_mode(raw_history_.current());
      if (!Game::Rules::is_chance_mode(child_mode)) {
        active_seat_ = Game::Rules::get_current_player(raw_history_.current());
      }
      applied_action = true;

      StateHistory* state_history = &raw_history_;
      if (canonical_sym_ != group::kIdentity) {
        calc_canonical_state_data();
        state_history = &pseudo_local_vars_.canonical_history;
      }

      if (expand(state_history, node, edge)) return;
    } else if (edge->state == Node::kMidExpansion) {
      if (multithreaded_) {
        node->cv().wait(lock, [edge] { return edge->state == Node::kExpanded; });
      }
    } else if (edge->state == Node::kPreExpanded) {
      edge->state = Node::kMidExpansion;
      lock.unlock();

      util::debug_assert(edge->child_index >= 0);
      Node* child = shared_data_->lookup_table.get_node(edge->child_index);
      search_path_.emplace_back(child, nullptr);
      int edge_count = edge->E;
      int child_count = child->stats().RN;  // not thread-safe but race-condition is benign
      if (edge_count < child_count) {
        short_circuit_backprop();
      } else {
        standard_backprop(false);
      }

      lock.lock();
      edge->state = Node::kExpanded;
      if (multithreaded_) {
        lock.unlock();
        node->cv().notify_all();
      }
      return;
    }
  }

  util::release_assert(edge->state == Node::kExpanded);
  Node* child = node->get_child(edge);
  if (child) {
    search_path_.emplace_back(child, nullptr);
    int edge_count = edge->E;
    int child_count = child->stats().RN;  // not thread-safe but race-condition is benign
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
    core::action_mode_t child_mode = Game::Rules::get_action_mode(raw_history_.current());
    if (!Game::Rules::is_chance_mode(child_mode)) {
      active_seat_ = Game::Rules::get_current_player(raw_history_.current());
    }
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
    LOG_INFO("{}{} {}", thread_id_whitespace(), __func__, search_path_str());
  }

  util::release_assert(!search_path_.empty());
  Node* last_node = search_path_.back().node;

  last_node->update_stats([&] {
    last_node->stats().VN++;  // thread-safe because executed under mutex
  });

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    Edge* edge = search_path_[i].edge;
    Node* node = search_path_[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().VN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path();
}

template <core::concepts::Game Game>
void SearchThread<Game>::undo_virtual_backprop() {
  // NOTE: this is not an exact undo of virtual_backprop(), since the search_path_ is modified in
  // between the two calls.
  profiler_.record(SearchThreadRegion::kUndoVirtualBackprop);

  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{}{} {}", thread_id_whitespace(), __func__, search_path_str());
  }

  util::release_assert(!search_path_.empty());

  for (int i = search_path_.size() - 1; i >= 0; --i) {
    Edge* edge = search_path_[i].edge;
    Node* node = search_path_[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E--;
      node->stats().VN--;  // thread-safe because executed under mutex
    });
  }
  validate_search_path();
}

template <core::concepts::Game Game>
inline void SearchThread<Game>::pure_backprop(const ValueArray& value) {
  profiler_.record(SearchThreadRegion::kPureBackprop);

  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{}{} {} {}", thread_id_whitespace(), __func__, search_path_str(),
             fmt::streamed(value.transpose()));
  }

  util::release_assert(!search_path_.empty());
  Node* last_node = search_path_.back().node;

  last_node->update_stats([&] {
    auto& stats = last_node->stats();  // thread-safe because executed under mutex
    stats.init_q(value, true);
    stats.RN++;
  });

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    Edge* edge = search_path_[i].edge;
    Node* node = search_path_[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().RN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path();
}

template <core::concepts::Game Game>
void SearchThread<Game>::standard_backprop(bool undo_virtual) {
  profiler_.record(SearchThreadRegion::kStandardBackprop);

  Node* last_node = search_path_.back().node;
  auto value = Game::GameResults::to_value_array(last_node->stable_data().VT);

  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{}{} {} {}", thread_id_whitespace(), __func__, search_path_str(),
             fmt::streamed(value.transpose()));
  }

  last_node->update_stats([&] {
    auto& stats = last_node->stats();  // thread-safe because executed under mutex
    stats.init_q(value, false);
    stats.RN++;
    stats.VN -= undo_virtual;
  });

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    Edge* edge = search_path_[i].edge;
    Node* node = search_path_[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E += !undo_virtual;
      auto& stats = node->stats();  // thread-safe because executed under mutex
      stats.RN++;
      stats.VN -= undo_virtual;
    });
  }
  validate_search_path();
}

template <core::concepts::Game Game>
void SearchThread<Game>::short_circuit_backprop() {
  profiler_.record(SearchThreadRegion::kShortCircuitBackprop);

  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{}{} {}", thread_id_whitespace(), __func__, search_path_str());
  }

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    Edge* edge = search_path_[i].edge;
    Node* node = search_path_[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().RN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path();
}

template <core::concepts::Game Game>
bool SearchThread<Game>::expand(StateHistory* history, Node* parent, Edge* edge) {
  profiler_.record(SearchThreadRegion::kExpand);

  LookupTable& lookup_table = shared_data_->lookup_table;
  MCTSKey mcts_key = Game::InputTensorizor::mcts_key(*history);

  // NOTE: we do a lookup_node() call here, and then later, inside init_node(), we do a
  // corresponding insert_node() call. This is analagous to:
  //
  // if key not in dict:
  //   ...
  //   dict[key] = value
  //
  // If there are multiple search threads, this represents a potential race-condition. The
  // straightforward solution is to hold a mutex during that entire sequence of operations. However,
  // this would hold the mutex for far too long.
  //
  // Instead, the below code carefully detects whether the race-condition has occurred, and if so,
  // keeps the first init_node() and "unwinds" the second one.
  node_pool_index_t child_index = lookup_table.lookup_node(mcts_key);

  bool is_new_node = child_index < 0;
  if (is_new_node) {
    child_index = lookup_table.alloc_node();
    Node* child = lookup_table.get_node(child_index);

    ValueTensor game_outcome;
    core::action_t last_action = edge->action;
    Game::Symmetries::apply(last_action, edge->sym, parent->action_mode());

    bool terminal = Game::Rules::is_terminal(
        history->current(), parent->stable_data().active_seat, last_action, game_outcome);

    if (terminal) {
      new (child) Node(&lookup_table, *history, game_outcome);
    } else {
      new (child) Node(&lookup_table, *history, active_seat_);
    }

    search_path_.emplace_back(child, nullptr);
    child->initialize_edges();
    bool do_virtual = !terminal && manager_params_->num_search_threads > 1;
    if (do_virtual) {
      virtual_backprop();
    }
    edge->child_index = init_node(history, child_index, child);
    if (child_index != edge->child_index) {
      // This means that we hit the race-condition described above. We need to "unwind" the second
      // init_node() call, and instead use the first one.
      //
      // Note that all the work done in constructing child is effectively discarded. We don't
      // need to explicit undo the alloc_node() call, as the memory will naturally be reclaimed
      // when the lookup_table is defragmented.
      search_path_.pop_back();
      if (do_virtual) {
        undo_virtual_backprop();
      }
      is_new_node = false;
      child_index = edge->child_index;
    } else {
      if (terminal) {
        pure_backprop(Game::GameResults::to_value_array(child->stable_data().VT));
      } else {
        standard_backprop(do_virtual);
      }
    }
  }

  if (!is_new_node) {
    // TODO: in this case, we should check to see if there are sister edges that point to the same
    // child. In this case, we can "slide" the visits and policy-mass from one edge to the other,
    // effectively pretending that we had merged the two edges from the beginning. This should
    // result in a more efficient search.
    //
    // We had something like this at some point, and for tic-tac-toe, it led to a significant
    // improvement. But that previous implementation was inefficient for large branching factors,
    // as it did the edge-merging up-front. This proposal only attempts edge-merges on-demand,
    // piggy-backing existing MCGS-key-lookups for minimal additional overhead.
    //
    // Some technical notes on this:
    //
    // - At a minimum we want to slide E and adjusted_base_prob, and then mark the edge as defunct,
    //   so that PUCT will not select it.
    // - We can easily mutex-protect the writes, by doing this under the parent's mutex. For the
    //   reads in ActionSelector, we can probably be unsafe. I think a reasonable order would be:
    //
    //   edge1->merged_edge_index = edge2_index;
    //   edge2->adjusted_base_prob += edge1->adjusted_base_prob;
    //   edge1->adjusted_base_prob = 0;
    //   edge2->E += edge1->E;
    //   edge1->E = 0;
    //
    //   We just have to reason carefully about the order of the reads in ActionSelector. Choosing
    //   which edge merges into which edge can also give us more control over possible races, as
    //   ActionSelector iterates over the edges in a specific order. More careful analysis is
    //   needed here.
    //
    //   Wherever we increment an edge->E, we can check, under the parent-mutex, if
    //   edge->merged_edge_index >= 0, and if so, increment the E of the merged edge instead, in
    //   order to make the writes thread-safe.
    edge->child_index = child_index;
  }

  std::unique_lock lock(parent->mutex());
  parent->update_child_expand_count();
  edge->state = Node::kExpanded;

  if (multithreaded_) {
    lock.unlock();
    parent->cv().notify_all();
  }
  return is_new_node;
}

template <core::concepts::Game Game>
std::string SearchThread<Game>::search_path_str() const {
  using Group = Game::SymmetryGroup;
  group::element_t cur_sym = Group::inverse(shared_data_->root_info.canonical_sym);
  std::string delim = Game::IO::action_delimiter();
  std::vector<std::string> vec;
  for (const Visitation& visitation : search_path_) {
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
    for (const Visitation& visitation : search_path_) {
      Edge* edge = visitation.edge;
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
  int n = node->stable_data().num_valid_actions;
  float chance_dist[n];
  for (int i = 0; i < n; i++) {
    chance_dist[i] = node->get_edge(i)->base_prob;
  }
  return util::Random::weighted_sample(chance_dist, chance_dist + n);
}

template <core::concepts::Game Game>
void SearchThread<Game>::print_action_selection_details(Node* node, const ActionSelector& selector,
                                                        int argmax_index) const {
  if (mcts::kEnableSearchDebug) {
    std::ostringstream ss;
    ss << thread_id_whitespace();

    core::seat_index_t seat = node->stable_data().active_seat;

    int n_actions = node->stable_data().num_valid_actions;

    ValueArray players;
    ValueArray nQ = node->stats().Q;
    ValueArray CP;
    for (int p = 0; p < kNumPlayers; ++p) {
      players(p) = p;
      CP(p) = p == seat;
    }

    static std::vector<std::string> player_columns = {"Seat", "Q", "CurP"};
    auto player_data = eigen_util::concatenate_columns(players, nQ, CP);

    eigen_util::PrintArrayFormatMap fmt_map1 {
      {"Seat", [&](float x) { return std::to_string(int(x)); }},
      {"CurP", [&](float x) { return std::string(x == seat ? "*" : ""); }},
    };

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

    static std::vector<std::string> action_columns = {
        "action", "P", "Q", "FPU", "PW", "PL", "E", "mE", "RN", "VN", "&ch", "PUCT", "argmax"};
    auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
        actions, P, Q, FPU, PW, PL, E, mE, RN, VN, child_addr, PUCT, argmax));

    eigen_util::PrintArrayFormatMap fmt_map2 {
      {"action", [&](float x) { return Game::IO::action_to_str(x, node->action_mode()); }},
      {"&ch", [](float x) { return x < 0 ? std::string() : std::to_string((int)x); }},
      {"argmax", [](float x) { return std::string(x == 0 ? "" : "*"); }},
    };

    std::stringstream ss2;
    eigen_util::print_array(ss2, action_data, action_columns, &fmt_map2);

    for (const std::string& line : util::splitlines(ss2.str())) {
      ss << line << break_plus_thread_id_whitespace();
    }

    LOG_INFO(ss.str());
  }
}

}  // namespace mcts
