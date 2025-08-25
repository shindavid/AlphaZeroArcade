#include "mcts/Node.hpp"

#include "util/CppUtil.hpp"

namespace mcts {

template <typename Traits>
void Node<Traits>::Stats::init_q(const ValueArray& value, bool pure) {
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

template <typename Traits>
void Node<Traits>::Stats::update_provable_bits(const player_bitset_t& all_actions_provably_winning,
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

template <typename Traits>
void Node<Traits>::write_results(const ManagerParams& params, group::element_t inv_sym,
                                 SearchResults& results) const {
  // This should only be called in contexts where the search-threads are inactive, so we do not need
  // to worry about thread-safety

  core::seat_index_t seat = this->stable_data().active_seat;
  DEBUG_ASSERT(seat >= 0 && seat < kNumPlayers);

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

  for (int i = 0; i < this->stable_data().num_valid_actions; i++) {
    const Edge* edge = this->get_edge(i);
    core::action_t action = edge->action;

    int count = edge->E;
    int modified_count = count;

    const Node* child = this->get_child(edge);
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
    RELEASE_ASSERT(stable_data.VT_valid);
    ValueArray VA = Game::GameResults::to_value_array(stable_data.VT);
    action_values(action) = VA(seat);
  }
}

template <typename Traits>
template <typename MutexProtectedFunc>
void Node<Traits>::update_stats(MutexProtectedFunc func) {
  mit::unique_lock lock(this->mutex());
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

  if (this->stable_data_.is_chance_node) {
    for (int i = 0; i < this->stable_data_.num_valid_actions; i++) {
      const Edge* edge = this->get_edge(i);
      const Node* child = this->get_child(edge);

      if (!child) {
        break;
      }
      const auto child_stats = child->stats_safe();  // make a copy
      Q_sum += child_stats.Q * edge->chance_prob;
      Q_sq_sum += child_stats.Q_sq * edge->chance_prob;
      N++;

      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
    }
    if (N == this->stable_data_.num_valid_actions) {
      lock.lock();

      stats_.Q = Q_sum;
      stats_.Q_sq = Q_sq_sum;
      stats_.provably_winning = all_provably_winning;
      stats_.provably_losing = all_provably_losing;
    }

  } else {
    core::seat_index_t seat = this->stable_data().active_seat;

    // provably winning/losing calculation
    bool cp_has_winning_move = false;
    int num_children = 0;

    bool skipped = false;
    for (int i = 0; i < this->stable_data().num_valid_actions; i++) {
      const Edge* edge = this->get_edge(i);
      const Node* child = this->get_child(edge);
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

    if (this->stable_data_.VT_valid) {
      ValueArray VA = Game::GameResults::to_value_array(this->stable_data_.VT);
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
                                cp_has_winning_move, this->stable_data_);

    if (N) {
      eigen_util::debug_assert_is_valid_prob_distr(stats_.Q);
    }
  }
}

template <typename Traits>
typename Node<Traits>::Stats Node<Traits>::stats_safe() const {
  // NOTE[dshin]: I attempted a version of this that attempted a lock-free read, resorting to a
  // the mutex only when a set dirty-bit was found on the copied stats. Contrary to my expectations,
  // this was slightly but clearly slower than the current version. I don't really understand why
  // this might be, but it's not worth investigating further at this time.
  mit::unique_lock lock(this->mutex());
  return stats_;
}

template <typename Traits>
template <typename PolicyTransformFunc>
void Node<Traits>::load_eval(NNEvaluation* eval, PolicyTransformFunc f) {
  int n = this->stable_data_.num_valid_actions;
  ValueTensor VT;

  LocalPolicyArray P_raw(n);
  LocalActionValueArray child_V(n);
  eval->load(VT, P_raw, child_V);

  LocalPolicyArray P_adjusted = P_raw;
  f(P_adjusted);

  this->stable_data_.VT = VT;
  this->stable_data_.VT_valid = true;

  // No need to worry about thread-safety when modifying edges or stats below, since no other
  // threads can access this node until after load_eval() returns
  for (int i = 0; i < n; ++i) {
    Edge* edge = this->get_edge(i);
    edge->policy_prior_prob = P_raw[i];
    edge->adjusted_base_prob = P_adjusted[i];
    edge->child_V_estimate = child_V[i];
  }

  ValueArray VA = Game::GameResults::to_value_array(VT);
  stats_.Q = VA;
  stats_.Q_sq = VA * VA;

  eigen_util::debug_assert_is_valid_prob_distr(VA);
}

template <typename Traits>
void Node<Traits>::validate_state() const {
  if (!IS_DEFINED(DEBUG_BUILD)) return;
  if (this->is_terminal()) return;

  mit::unique_lock lock(this->mutex());

  int N = 1;
  for (int i = 0; i < this->stable_data_.num_valid_actions; ++i) {
    auto edge = this->get_edge(i);
    N += edge->E;
    DEBUG_ASSERT(edge->E >= 0);
  }

  const auto stats_copy = stats();  // thread-safe because we hold the mutex
  lock.unlock();

  DEBUG_ASSERT(N == stats_copy.RN + stats_copy.VN, "[{}] {} != {} + {}", (void*)this, N,
               stats_copy.RN, stats_copy.VN);
  DEBUG_ASSERT(stats_copy.RN >= 0);
  DEBUG_ASSERT(stats_copy.VN >= 0);
}

}  // namespace mcts
