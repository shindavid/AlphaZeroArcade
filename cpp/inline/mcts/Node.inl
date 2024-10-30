#include <mcts/Node.hpp>

#include <util/CppUtil.hpp>
#include <util/LoggingUtil.hpp>

namespace mcts {

template <core::concepts::Game Game>
inline Node<Game>::stable_data_t::stable_data_t(const StateHistory& history)
    : StateData(history.current()) {
  VT.setZero();  // to be set lazily
  VT_valid = false;
  valid_action_mask = Game::Rules::get_legal_moves(history);
  num_valid_actions = valid_action_mask.count();
  current_player = Game::Rules::get_current_player(history.current());
  terminal = false;
}

template <core::concepts::Game Game>
inline Node<Game>::stable_data_t::stable_data_t(const StateHistory& history,
                                                const ValueTensor& game_outcome)
    : StateData(history.current()) {
  VT = game_outcome;
  VT_valid = true;
  num_valid_actions = 0;
  current_player = -1;
  terminal = true;
}

template <core::concepts::Game Game>
void Node<Game>::stats_t::init_q(const ValueArray& value, bool pure) {
  Q = value;
  Q_sq = value * value;
  if (pure) {
    Q_lower_bound = value;
    Q_upper_bound = value;
  } else {
    Q_lower_bound.setConstant(Game::GameResults::kMinValue);
    Q_upper_bound.setConstant(Game::GameResults::kMaxValue);
  }

  eigen_util::debug_assert_is_valid_prob_distr(Q);
}

template <core::concepts::Game Game>
Node<Game>::LookupTable::Defragmenter::Defragmenter(LookupTable* table)
    : table_(table),
      node_bitset_(table->node_pool_.size()),
      edge_bitset_(table->edge_pool_.size()) {}

template <core::concepts::Game Game>
void Node<Game>::LookupTable::Defragmenter::scan(node_pool_index_t n) {
  if (n < 0 || node_bitset_[n]) return;

  node_bitset_[n] = true;
  Node* node = &table_->node_pool_[n];
  if (!node->edges_initialized()) return;

  edge_pool_index_t first_edge_index = node->get_first_edge_index();
  int n_edges = node->stable_data().num_valid_actions;

  edge_bitset_.set(first_edge_index, n_edges, true);
  for (int e = 0; e < n_edges; ++e) {
    scan(node->get_edge(e)->child_index);
  }
}

template <core::concepts::Game Game>
void Node<Game>::LookupTable::Defragmenter::prepare() {
  init_remapping(node_index_remappings_, node_bitset_);
  init_remapping(edge_index_remappings_, edge_bitset_);
}

template <core::concepts::Game Game>
void Node<Game>::LookupTable::Defragmenter::remap(node_pool_index_t& n) {
  bitset_t processed_nodes(table_->node_pool_.size());
  remap_helper(n, processed_nodes);
  n = node_index_remappings_[n];
  util::debug_assert(processed_nodes == node_bitset_);
}

template <core::concepts::Game Game>
void Node<Game>::LookupTable::Defragmenter::defrag() {
  table_->node_pool_.defragment(node_bitset_);
  table_->edge_pool_.defragment(edge_bitset_);

  for (auto it = table_->map_.begin(); it != table_->map_.end();) {
    if (!node_bitset_[it->second]) {
      it = table_->map_.erase(it);
    } else {
      it->second = node_index_remappings_[it->second];
      ++it;
    }
  }
}

template <core::concepts::Game Game>
void Node<Game>::LookupTable::Defragmenter::remap_helper(node_pool_index_t n,
                                                         bitset_t& processed_nodes) {
  if (processed_nodes[n]) return;

  processed_nodes[n] = true;
  Node* node = &table_->node_pool_[n];
  if (!node->edges_initialized()) return;

  edge_pool_index_t first_edge_index = node->get_first_edge_index();
  int n_edges = node->stable_data().num_valid_actions;

  for (int e = 0; e < n_edges; ++e) {
    edge_t* edge = node->get_edge(e);
    if (edge->child_index < 0) continue;
    remap_helper(edge->child_index, processed_nodes);
    edge->child_index = node_index_remappings_[edge->child_index];
  }

  node->set_first_edge_index(edge_index_remappings_[first_edge_index]);
}

template <core::concepts::Game Game>
void Node<Game>::LookupTable::Defragmenter::init_remapping(index_vec_t& remappings,
                                                           bitset_t& bitset) {
  remappings.resize(bitset.size());
  for (int i = 0; i < (int)bitset.size(); ++i) {
    remappings[i] = -1;
  }

  auto i = bitset.find_first();
  int k = 0;
  while (i != bitset_t::npos) {
    remappings[i] = k++;
    i = bitset.find_next(i);
  }
}

template <core::concepts::Game Game>
Node<Game>::LookupTable::LookupTable(bool multithreaded_mode)
    : mutex_pool_(multithreaded_mode ? kDefaultMutexPoolSize : 1),
      cv_pool_(multithreaded_mode ? kDefaultMutexPoolSize : 1) {}

template <core::concepts::Game Game>
void Node<Game>::LookupTable::clear() {
  map_.clear();
  edge_pool_.clear();
  node_pool_.clear();
}

template <core::concepts::Game Game>
void Node<Game>::LookupTable::defragment(node_pool_index_t& root_index) {
  Defragmenter defragmenter(this);
  defragmenter.scan(root_index);
  defragmenter.prepare();
  defragmenter.remap(root_index);
  defragmenter.defrag();
}

template <core::concepts::Game Game>
void Node<Game>::LookupTable::insert_node(const MCTSKey& key, node_pool_index_t value) {
  std::lock_guard lock(map_mutex_);
  map_[key] = value;
}

template <core::concepts::Game Game>
typename Node<Game>::node_pool_index_t Node<Game>::LookupTable::lookup_node(
    const MCTSKey& key) const {
  std::lock_guard lock(map_mutex_);
  auto it = map_.find(key);
  if (it == map_.end()) {
    return -1;
  }
  return it->second;
}

template <core::concepts::Game Game>
int Node<Game>::LookupTable::get_random_mutex_id() const {
  return util::Random::uniform_sample(0, (int)mutex_pool_.size());
}

template <core::concepts::Game Game>
Node<Game>::Node(LookupTable* table, const StateHistory& history)
    : stable_data_(history),
      lookup_table_(table),
      mutex_id_(table->get_random_mutex_id()) {}

template <core::concepts::Game Game>
Node<Game>::Node(LookupTable* table, const StateHistory& history, const ValueTensor& game_outcome)
    : stable_data_(history, game_outcome),
      lookup_table_(table),
      mutex_id_(table->get_random_mutex_id()) {}

template <core::concepts::Game Game>
void Node<Game>::write_results(const ManagerParams& params, group::element_t inv_sym,
                               SearchResults& results) const {
  // This should only be called in contexts where the search-threads are inactive, so we do not need
  // to worry about thread-safety

  core::seat_index_t cp = stable_data().current_player;

  auto& counts = results.counts;
  auto& action_values = results.action_values;
  auto& Q = results.Q;
  auto& Q_sq = results.Q_sq;

  counts.setZero();
  action_values.setZero();
  Q.setZero();
  Q_sq.setZero();

  for (int i = 0; i < stable_data().num_valid_actions; i++) {
    const edge_t* edge = get_edge(i);
    core::action_t action = edge->action;

    const Node* child = get_child(edge);
    if (!child) continue;

    const auto& stats = child->stats();

    if (!edge->eliminated) {
      counts(action) = edge->E;
      Q(action) = stats.Q(cp);
      Q_sq(action) = stats.Q_sq(cp);
    }

    const auto& stable_data = child->stable_data();
    util::release_assert(stable_data.VT_valid);
    ValueArray VA = Game::GameResults::to_value_array(stable_data.VT);
    action_values(action) = VA(cp);
  }
}

template <core::concepts::Game Game>
template <typename MutexProtectedFunc>
bool Node<Game>::update_stats(MutexProtectedFunc func) {
  std::unique_lock lock(mutex());
  func();
  lock.unlock();

  int n_actions = stable_data_.num_valid_actions;

  float best_bounds[2];
  set_best_bounds(best_bounds);

  bool eliminated_actions[n_actions] = {};
  set_eliminated_actions(eliminated_actions, best_bounds);

  ValueArray Q_lower_bound;
  ValueArray Q_upper_bound;
  compute_Q_bounds(Q_lower_bound, Q_upper_bound, eliminated_actions, best_bounds);

  ValueArray Q;
  ValueArray Q_sq;
  compute_Q_values(Q, Q_sq, Q_lower_bound, Q_upper_bound, eliminated_actions);

  // Now we lock the mutex, and copy the calculated information into the node
  lock.lock();

  // Mask edges that are eliminated
  bool fresh_elimination = false;
  for (int i = 0; i < n_actions; i++) {
    edge_t* edge = get_edge(i);
    fresh_elimination |= !edge->eliminated && eliminated_actions[i];
    edge->eliminated = eliminated_actions[i];
  }

  // Copy Q info
  stats_.Q = Q;
  stats_.Q_sq = Q_sq;
  if (!is_terminal()) {
    stats_.Q_lower_bound = Q_lower_bound;
    stats_.Q_upper_bound = Q_upper_bound;
  }

  return fresh_elimination;
}

// NOTE: this can be switched to use binary search if we'd like
template <core::concepts::Game Game>
typename Node<Game>::node_pool_index_t Node<Game>::lookup_child_by_action(
    core::action_t action) const {
  int i = 0;
  for (core::action_t a : bitset_util::on_indices(stable_data_.valid_action_mask)) {
    if (a == action) {
      return get_edge(i)->child_index;
    }
    ++i;
  }
  return -1;
}

template <core::concepts::Game Game>
void Node<Game>::initialize_edges() {
  int n_edges = stable_data_.num_valid_actions;
  if (n_edges == 0) return;
  first_edge_index_ = lookup_table_->alloc_edges(n_edges);

  int i = 0;
  for (core::action_t action : bitset_util::on_indices(stable_data_.valid_action_mask)) {
    edge_t* edge = get_edge(i);
    new (edge) edge_t();
    edge->action = action;
    i++;
  }
}

template <core::concepts::Game Game>
template <typename PolicyTransformFunc>
void Node<Game>::load_eval(NNEvaluation* eval, PolicyTransformFunc f) {
  int n = stable_data_.num_valid_actions;
  ValueTensor VT;

  LocalPolicyArray P_raw(n);
  LocalActionValueArray child_V(n);
  eval->load(VT, P_raw, child_V);

  LocalPolicyArray P_adjusted = P_raw;
  f(P_adjusted);

  stable_data_.VT = VT;
  stable_data_.VT_valid = true;

  for (int i = 0; i < n; ++i) {
    edge_t* edge = get_edge(i);
    edge->raw_policy_prior = P_raw[i];
    edge->adjusted_policy_prior = P_adjusted[i];
    edge->child_V_estimate = child_V[i];
  }

  ValueArray VA = Game::GameResults::to_value_array(VT);
  stats_.Q = VA;
  stats_.Q_sq = VA * VA;

  eigen_util::debug_assert_is_valid_prob_distr(VA);
}

template <core::concepts::Game Game>
bool Node<Game>::all_children_edges_initialized() const {
  if (stable_data_.num_valid_actions == 0) return true;
  if (first_edge_index_ == -1) return false;
  for (int j = 0; j < stable_data_.num_valid_actions; ++j) {
    if (get_edge(j)->state != kExpanded) return false;
  }
  return true;
}

template <core::concepts::Game Game>
typename Node<Game>::edge_t* Node<Game>::get_edge(int i) const {
  util::debug_assert(first_edge_index_ != -1);
  return lookup_table_->get_edge(first_edge_index_ + i);
}

template <core::concepts::Game Game>
Node<Game>* Node<Game>::get_child(const edge_t* edge) const {
  if (edge->child_index < 0) return nullptr;
  return lookup_table_->get_node(edge->child_index);
}

template <core::concepts::Game Game>
void Node<Game>::update_child_expand_count(int n) {
  child_expand_count_ += n;
  util::debug_assert(child_expand_count_ <= stable_data_.num_valid_actions);
  if (child_expand_count_ < stable_data_.num_valid_actions) return;

  // all children have been expanded, check for triviality
  if (child_expand_count_ == 0) return;
  node_pool_index_t first_child_index = get_edge(0)->child_index;
  for (int i = 1; i < stable_data_.num_valid_actions; ++i) {
    if (get_edge(i)->child_index != first_child_index) return;
  }

  trivial_ = true;
}

template <core::concepts::Game Game>
void Node<Game>::validate_state() const {
  if (!IS_MACRO_ENABLED(DEBUG_BUILD)) return;
  if (is_terminal()) return;

  std::unique_lock lock(mutex());

  int N = 1;
  for (int i = 0; i < stable_data_.num_valid_actions; ++i) {
    auto edge = get_edge(i);
    N += edge->E;
    util::debug_assert(edge->E >= 0);
  }

  util::debug_assert(N == stats_.RN + stats_.VN, "[%p] %d != %d + %d", this, N, stats_.RN,
                     stats_.VN);
  util::debug_assert(stats_.RN >= 0);
  util::debug_assert(stats_.VN >= 0);
}


template <core::concepts::Game Game>
void Node<Game>::set_best_bounds(float* best_bounds) const {
  best_bounds[0] = Game::GameResults::kMinValue;
  best_bounds[1] = Game::GameResults::kMinValue;

  int n_actions = stable_data_.num_valid_actions;
  core::seat_index_t cp = stable_data_.current_player;

  bool best_bounds_set = false;
  for (int i = 0; i < n_actions; i++) {
    const edge_t* edge = get_edge(i);
    const Node* child = get_child(edge);
    if (!child) continue;

    const auto& child_stats = child->stats();
    float ql = child_stats.Q_lower_bound(cp);
    float qu = child_stats.Q_upper_bound(cp);

    best_bounds_set = true;
    if (ql > best_bounds[0]) {
      best_bounds[0] = ql;
      best_bounds[1] = qu;
    } else if (ql == best_bounds[0]) {
      best_bounds[1] = std::max(best_bounds[1], qu);
    }
  }

  if (!best_bounds_set) {
    best_bounds[1] = Game::GameResults::kMaxValue;
  }
}

template <core::concepts::Game Game>
void Node<Game>::set_eliminated_actions(bool* eliminated_actions, const float* best_bounds) const {
  int n_actions = stable_data_.num_valid_actions;
  core::seat_index_t cp = stable_data_.current_player;

  for (int i = 0; i < n_actions; i++) {
    const edge_t* edge = get_edge(i);
    const Node* child = get_child(edge);
    if (!child) {
      // if best_bounds[0] is kMaxValue, then the position has been proven to be winning, and we
      // can eliminate not-yet-expanded actions.
      eliminated_actions[i] = best_bounds[0] == Game::GameResults::kMaxValue;
      continue;
    }

    const auto& child_stats = child->stats();
    float ql = child_stats.Q_lower_bound(cp);
    float qu = child_stats.Q_upper_bound(cp);

    // The first condition is to prevent self-elimination
    eliminated_actions[i] = (ql < best_bounds[1] && qu <= best_bounds[0]);
  }
}

template <core::concepts::Game Game>
void Node<Game>::compute_Q_bounds(ValueArray& Q_lower_bound, ValueArray& Q_upper_bound,
                                  const bool* eliminated_actions, const float* best_bounds) const {
  int n_actions = stable_data_.num_valid_actions;
  core::seat_index_t cp = stable_data_.current_player;

  Q_lower_bound.setConstant(Game::GameResults::kMaxValue);
  Q_upper_bound.setConstant(Game::GameResults::kMinValue);

  // update Q bounds based only on non-eliminated children
  for (int i = 0; i < n_actions; i++) {
    if (eliminated_actions[i]) continue;

    const edge_t* edge = get_edge(i);
    const Node* child = get_child(edge);
    if (!child) {
      Q_lower_bound.setConstant(Game::GameResults::kMinValue);
      Q_upper_bound.setConstant(Game::GameResults::kMaxValue);
      break;
    }

    const auto& child_stats = child->stats();
    Q_lower_bound = Q_lower_bound.cwiseMin(child_stats.Q_lower_bound);
    Q_upper_bound = Q_upper_bound.cwiseMax(child_stats.Q_upper_bound);
  }

  // cp is the current player, and so can choose the best possible action for herself
  Q_lower_bound(cp) = best_bounds[0];

  // NOTE: the below assumes that the game is zero-sum, with the sum of Q values being 1. This
  // assumption should be relaxed, and the Game concept API should be extended to specify whether
  // this assumption holds.
  //
  // The below logic recognizes that if cp can force a Q-lower-bound of 0.4 for herself, then the
  // other players can obtain at best a Q-upper-bound of 0.6.
  //
  // It can probably be made more powerful by looking at Q-lower-bounds across all players: if
  // Player 1 has a Q-lower-bound of 0.4, and Player 2 has a Q-lower-bound of 0.1, then Player 3
  // can obtain a Q of at most 1.0 - 0.4 - 0.1 = 0.5.
  for (core::seat_index_t p = 0; p < kNumPlayers; ++p) {
    if (p == cp) continue;
    Q_upper_bound(p) = std::min(1.0f - Q_lower_bound(cp), Q_upper_bound(p));
  }
}

template <core::concepts::Game Game>
void Node<Game>::compute_Q_values(ValueArray& Q, ValueArray& Q_sq, const ValueArray& Q_lower_bound,
                        const ValueArray& Q_upper_bound, const bool* eliminated_actions) const {
  if ((Q_lower_bound == Q_upper_bound).all()) {
    Q = Q_lower_bound;
    Q_sq = Q * Q;
    eigen_util::debug_assert_is_valid_prob_distr(Q);
    return;
  }

  int n_actions = stable_data_.num_valid_actions;

  ValueArray Q_sum;
  ValueArray Q_sq_sum;
  Q_sum.setZero();
  Q_sq_sum.setZero();
  int N = 0;

  // update Q based on non-eliminated children
  for (int i = 0; i < n_actions; i++) {
    if (eliminated_actions[i]) continue;

    const edge_t* edge = get_edge(i);
    const Node* child = get_child(edge);
    if (!child) {
      continue;
    }

    const auto& child_stats = child->stats();
    if (child_stats.RN > 0) {
      int e = edge->E;
      N += e;
      Q_sum += child_stats.Q * e;
      Q_sq_sum += child_stats.Q_sq * e;
    }
  }


  if (stable_data_.VT_valid) {
    ValueArray VA = Game::GameResults::to_value_array(stable_data_.VT);

    // TODO: cap VA by Q_lower_bound and Q_upper_bound. Need to give some thought on how to do
    // this properly, given potential zero-sum property.

    Q_sum += VA;
    Q_sq_sum += VA * VA;
    N++;
  }

  if (N) {
    Q = Q_sum / N;
    Q_sq = Q_sq_sum / N;
    eigen_util::debug_assert_is_valid_prob_distr(Q);
  } else {
    Q.setZero();
    Q_sq.setZero();
  }
}

}  // namespace mcts
