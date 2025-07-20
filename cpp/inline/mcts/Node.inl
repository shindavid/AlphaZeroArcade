#include <mcts/Node.hpp>

#include <util/CppUtil.hpp>
#include <util/LoggingUtil.hpp>

namespace mcts {

template <core::concepts::Game Game>
inline Node<Game>::StableData::StableData(const StateHistory& history,
                                                core::seat_index_t as)
    : StateData(history.current()) {
  VT.setZero();  // to be set lazily
  VT_valid = false;
  valid_action_mask = Game::Rules::get_legal_moves(history);
  num_valid_actions = valid_action_mask.count();
  action_mode = Game::Rules::get_action_mode(history.current());
  is_chance_node = Game::Rules::is_chance_mode(action_mode);
  active_seat = as;
  terminal = false;
}

template <core::concepts::Game Game>
inline Node<Game>::StableData::StableData(const StateHistory& history,
                                                const ValueTensor& game_outcome)
    : StateData(history.current()) {
  VT = game_outcome;
  VT_valid = true;
  num_valid_actions = 0;
  action_mode = -1;
  active_seat = -1;
  terminal = true;
  is_chance_node = false;
}

template <core::concepts::Game Game>
void Node<Game>::Stats::init_q(const ValueArray& value, bool pure) {
  Q = value;
  Q_sq = value * value;
  if (pure) {
    for (int p = 0; p < kNumPlayers; ++p) {
      provably_winning[p] = value(p) >= Game::GameResults::kMaxValue;
      provably_losing[p] = value(p) <= Game::GameResults::kMinValue;
    }
  }

  eigen_util::debug_assert_is_valid_prob_distr(Q);
}

template <core::concepts::Game Game>
void Node<Game>::Stats::update_provable_bits(const player_bitset_t& all_actions_provably_winning,
                                               const player_bitset_t& all_actions_provably_losing,
                                               int num_expanded_children, bool cp_has_winning_move,
                                               const StableData& sdata) {
  int num_valid_actions = sdata.num_valid_actions;
  core::seat_index_t seat = sdata.active_seat;

  if (num_valid_actions == 0) {
    // terminal state, provably_winning/losing should already be set
  } else if (cp_has_winning_move) {
    provably_winning[seat] = true;
    provably_losing.set();
    provably_losing[seat] = false;
  } else if (num_expanded_children == num_valid_actions) {
    provably_winning = all_actions_provably_winning;
    provably_losing = all_actions_provably_losing;
  }
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
    Edge* edge = node->get_edge(e);
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
Node<Game>::LookupTable::LookupTable(mcts::mutex_vec_sptr_t mutex_pool)
    : mutex_pool_(mutex_pool)
    , mutex_pool_size_(mutex_pool->size()) {}

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
typename Node<Game>::node_pool_index_t
Node<Game>::LookupTable::insert_node(const MCTSKey& key, node_pool_index_t value, bool overwrite) {
  mit::lock_guard lock(map_mutex_);
  if (overwrite) {
    map_[key] = value;
    return value;
  } else {
    auto result = map_.emplace(key, value);
    return result.first->second;
  }
}

template <core::concepts::Game Game>
typename Node<Game>::node_pool_index_t Node<Game>::LookupTable::lookup_node(
    const MCTSKey& key) const {
  mit::lock_guard lock(map_mutex_);
  auto it = map_.find(key);
  if (it == map_.end()) {
    return -1;
  }
  return it->second;
}

template <core::concepts::Game Game>
int Node<Game>::LookupTable::get_random_mutex_id() const {
  return mutex_pool_size_ == 1 ? 0 : util::Random::uniform_sample(0, mutex_pool_size_);
}

template <core::concepts::Game Game>
mit::mutex& Node<Game>::LookupTable::get_mutex(int mutex_id) {
  return (*mutex_pool_)[mutex_id];
}

template <core::concepts::Game Game>
Node<Game>::Node(LookupTable* table, const StateHistory& history, core::seat_index_t active_seat)
    : stable_data_(history, active_seat),
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

  core::seat_index_t seat = stable_data().active_seat;
  util::debug_assert(seat >= 0 && seat < kNumPlayers);

  auto& counts = results.counts;
  auto& action_values = results.action_values;
  auto& Q = results.Q;
  auto& Q_sq = results.Q_sq;

  counts.setZero();
  action_values.setZero();
  Q.setZero();
  Q_sq.setZero();

  const auto& parent_stats = this->stats();  // thread-safe because single-threaded here

  bool provably_winning = parent_stats.provably_winning[seat];
  bool provably_losing = parent_stats.provably_losing[seat];

  for (int i = 0; i < stable_data().num_valid_actions; i++) {
    const Edge* edge = get_edge(i);
    core::action_t action = edge->action;

    int count = edge->E;
    int modified_count = count;

    const Node* child = get_child(edge);
    if (!child) continue;

    // not actually unsafe since single-threaded
    const auto& child_stats = child->stats();  // thread-safe because single-threaded here
    if (params.avoid_proven_losers && !provably_losing && child_stats.provably_losing[seat]) {
      modified_count = 0;
    } else if (params.exploit_proven_winners && provably_winning &&
               !child_stats.provably_winning[seat]) {
      modified_count = 0;
    }

    if (modified_count) {
      counts(action) = modified_count;
      Q(action) = child_stats.Q(seat);
      Q_sq(action) = child_stats.Q_sq(seat);
    }

    const auto& stable_data = child->stable_data();
    util::release_assert(stable_data.VT_valid);
    ValueArray VA = Game::GameResults::to_value_array(stable_data.VT);
    action_values(action) = VA(seat);
  }
}

template <core::concepts::Game Game>
template <typename MutexProtectedFunc>
void Node<Game>::update_stats(MutexProtectedFunc func) {
  mit::unique_lock lock(mutex());
  func();
  lock.unlock();

  ValueArray Q_sum;
  ValueArray Q_sq_sum;
  Q_sum.setZero();
  Q_sq_sum.setZero();
  int N = 0;

  player_bitset_t all_provably_winning;
  player_bitset_t all_provably_losing;
  all_provably_winning.set();
  all_provably_losing.set();

  if (stable_data_.is_chance_node) {
    for (int i = 0; i < stable_data_.num_valid_actions; i++) {
      const Edge* edge = get_edge(i);
      const Node* child = get_child(edge);

      if (!child) {
        break;
      }
      const auto child_stats = child->stats_safe();  // make a copy
      Q_sum += child_stats.Q * edge->base_prob;
      Q_sq_sum += child_stats.Q_sq * edge->base_prob;
      N++;

      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
    }
    if (N == stable_data_.num_valid_actions) {
      lock.lock();

      stats_.Q = Q_sum;
      stats_.Q_sq = Q_sq_sum;
      stats_.provably_winning = all_provably_winning;
      stats_.provably_losing = all_provably_losing;
    }

  } else {
    core::seat_index_t seat = stable_data().active_seat;

    // provably winning/losing calculation
    bool cp_has_winning_move = false;
    int num_children = 0;

    bool skipped = false;
    for (int i = 0; i < stable_data().num_valid_actions; i++) {
      const Edge* edge = get_edge(i);
      const Node* child = get_child(edge);
      if (!child) {
        skipped = true;
        continue;
      }
      const auto child_stats = child->stats_safe();  // make a copy
      if (child_stats.RN > 0) {
        int e = edge->E;
        N += e;
        Q_sum += child_stats.Q * e;
        Q_sq_sum += child_stats.Q_sq * e;
      }

      cp_has_winning_move |= child_stats.provably_winning[seat];
      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
      num_children++;
    }

    if (skipped) {
      all_provably_winning.reset();
      all_provably_losing.reset();
    }

    if (stable_data_.VT_valid) {
      ValueArray VA = Game::GameResults::to_value_array(stable_data_.VT);
      Q_sum += VA;
      Q_sq_sum += VA * VA;
      N++;

      eigen_util::debug_assert_is_valid_prob_distr(VA);
    }

    auto Q = N ? (Q_sum / N) : Q_sum;
    auto Q_sq = N ? (Q_sq_sum / N) : Q_sq_sum;

    lock.lock();

    stats_.Q = Q;
    stats_.Q_sq = Q_sq;
    stats_.update_provable_bits(all_provably_winning, all_provably_losing, num_children,
                                cp_has_winning_move, stable_data_);

    if (N) {
      eigen_util::debug_assert_is_valid_prob_distr(stats_.Q);
    }
  }
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
typename Node<Game>::Stats Node<Game>::stats_safe() const {
  // NOTE[dshin]: I attempted a version of this that attempted a lock-free read, resorting to a
  // the mutex only when a set dirty-bit was found on the copied stats. Contrary to my expectations,
  // this was slightly but clearly slower than the current version. I don't really understand why
  // this might be, but it's not worth investigating further at this time.
  mit::unique_lock lock(mutex());
  return stats_;
}

template <core::concepts::Game Game>
void Node<Game>::initialize_edges() {
  int n_edges = stable_data_.num_valid_actions;
  if (n_edges == 0) return;
  first_edge_index_ = lookup_table_->alloc_edges(n_edges);

  int i = 0;
  for (core::action_t action : bitset_util::on_indices(stable_data_.valid_action_mask)) {
    Edge* edge = get_edge(i);
    new (edge) Edge();
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

  // No need to worry about thread-safety when modifying edges or stats below, since no other
  // threads can access this node until after load_eval() returns
  for (int i = 0; i < n; ++i) {
    Edge* edge = get_edge(i);
    edge->base_prob = P_raw[i];
    edge->adjusted_base_prob = P_adjusted[i];
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
typename Node<Game>::Edge* Node<Game>::get_edge(int i) const {
  util::debug_assert(first_edge_index_ != -1);
  return lookup_table_->get_edge(first_edge_index_ + i);
}

template <core::concepts::Game Game>
Node<Game>* Node<Game>::get_child(const Edge* edge) const {
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
  if (!IS_DEFINED(DEBUG_BUILD)) return;
  if (is_terminal()) return;

  mit::unique_lock lock(mutex());

  int N = 1;
  for (int i = 0; i < stable_data_.num_valid_actions; ++i) {
    auto edge = get_edge(i);
    N += edge->E;
    util::debug_assert(edge->E >= 0);
  }

  const auto stats_copy = stats();  // thread-safe because we hold the mutex
  lock.unlock();

  util::debug_assert(N == stats_copy.RN + stats_copy.VN, "[{}] {} != {} + {}", (void*)this, N,
                     stats_copy.RN, stats_copy.VN);
  util::debug_assert(stats_copy.RN >= 0);
  util::debug_assert(stats_copy.VN >= 0);
}

}  // namespace mcts
