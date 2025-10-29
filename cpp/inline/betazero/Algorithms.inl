#include "betazero/Algorithms.hpp"

#include "util/Asserts.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <cmath>
#include <limits>

namespace beta0 {

namespace detail {

// Your alpha, computed without forming p or q:
inline double alpha_from_thetas(double theta_old, double omega_sq_old, double theta_new,
                                double omega_sq_new, double theta_i, double omega_sq_i) {
  const double z_old = (theta_old - theta_i) /
                       std::sqrt(omega_sq_old + omega_sq_i);
  const double z_new = (theta_new - theta_i) /
                       std::sqrt(omega_sq_new + omega_sq_i);

  return math::normal_cdf_logit_diff(z_new, z_old);
}

}  // namespace detail

template <typename T>
void check_values(const T& t, int line) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;

  auto data = t.data();
  int n = t.size();
  for (int i = 0; i < n; ++i) {
    RELEASE_ASSERT(data[i] >= 0.f, "invalid value ({}) at line {}", data[i], line);
    RELEASE_ASSERT(data[i] <= 1.f, "invalid value ({}) at line {}", data[i], line);
  }
}

template <search::concepts::Traits Traits, typename Derived>
template <typename MutexProtectedFunc>
void AlgorithmsBase<Traits, Derived>::Backpropagator::run(Node* node, Edge* edge,
                                                          MutexProtectedFunc&& func) {
  mit::unique_lock lock(node->mutex());
  // LOG_INFO("Backpropagator::run()");
  func();
  if (!snapshots_set_) {
    if (edge) {
      // This corresponds to the short-circuit-backprop or undo-virtual-backprop case.
      update_edge_snapshots(node, edge, true);
    } else {
      last_child_Qbeta_snapshot_ = node->stats().Qbeta;
      last_child_W_snapshot_ = node->stats().W;
      snapshots_set_ = true;
      // LOG_INFO("Updating snapshots @{}:", __LINE__);
      // LOG_INFO("last_child_Qbeta_snapshot_ = {}",
      //          fmt::streamed(last_child_Qbeta_snapshot_.transpose()));
      // LOG_INFO("last_child_W_snapshot_ = {}", fmt::streamed(last_child_W_snapshot_.transpose()));
      return;
    }
  } else {
    RELEASE_ASSERT(edge, "unexpected null edge on subsequent Backpropagator::run() call");
  }

  RELEASE_ASSERT(snapshots_set_);
  NodeStats stats = node->stats();  // copy

  ValueArray tmp_last_child_Qbeta_snapshot = stats.Qbeta;
  ValueArray tmp_last_child_W_snapshot = stats.W;

  int num_valid_actions = node->stable_data().num_valid_actions;
  LocalPolicyArray pi_arr(num_valid_actions);
  for (int i = 0; i < num_valid_actions; i++) {
    const Edge* child_edge = lookup_table_.get_edge(node, i);
    pi_arr(i) = child_edge->policy_posterior_prob;
  }
  lock.unlock();

  Derived::update_stats(stats, pi_arr, last_child_Qbeta_snapshot_, last_child_W_snapshot_, node,
                        edge, lookup_table_);

  // LOG_INFO("Update result:");
  // LOG_INFO("Qbeta = {} -> {}", fmt::streamed(tmp_last_child_Qbeta_snapshot.transpose()),
  //          fmt::streamed(stats.Qbeta.transpose()));
  // LOG_INFO("W = {} -> {}", fmt::streamed(tmp_last_child_W_snapshot.transpose()),
  //          fmt::streamed(stats.W.transpose()));

  lock.lock();

  update_edge_snapshots(node, edge);
  last_child_Qbeta_snapshot_ = tmp_last_child_Qbeta_snapshot;
  last_child_W_snapshot_ = tmp_last_child_W_snapshot;
  // LOG_INFO("Updating snapshots @{}:", __LINE__);
  // LOG_INFO("last_child_Qbeta_snapshot_ = {}",
  //          fmt::streamed(last_child_Qbeta_snapshot_.transpose()));
  // LOG_INFO("last_child_W_snapshot_ = {}", fmt::streamed(last_child_W_snapshot_.transpose()));
  for (int i = 0; i < num_valid_actions; i++) {
    Edge* child_edge = lookup_table_.get_edge(node, i);
    child_edge->policy_posterior_prob = pi_arr[i];
  }

  // Carefully copy fields of stats back to node->stats()
  // We don't copy counts, which may have been updated by other threads.
  int RN = node->stats().RN;
  int VN = node->stats().VN;
  node->stats() = stats;
  node->stats().RN = RN;
  node->stats().VN = VN;
  lock.unlock();
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::Backpropagator::update_edge_snapshots(
  Node* parent, Edge* edge, bool load_edge_snapshots_before_update) {
  // Like Algorithsm::init_edge_from_child(), but assumes parent mutex is already held

  const Node* child = lookup_table_.get_node(edge->child_index);
  if (!child) return;

  if (&parent->mutex() == &child->mutex()) {
    // parent and child are sharing the same mutex (this happens because mutexes are allocated from
    // a pool).
    const NodeStats& child_stats = child->stats();  // do not acquire child mutex
    update_edge_snapshots_helper(child_stats, edge, load_edge_snapshots_before_update);
  } else {
    NodeStats child_stats = child->stats_safe();  // acquire child mutex and make a copy
    update_edge_snapshots_helper(child_stats, edge, load_edge_snapshots_before_update);
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::Backpropagator::update_edge_snapshots_helper(
  const NodeStats& child_stats, Edge* edge, bool load_edge_snapshots_before_update) {
  if (load_edge_snapshots_before_update) {
    last_child_Qbeta_snapshot_ = edge->child_Qbeta_snapshot;
    last_child_W_snapshot_ = edge->child_W_snapshot;
    snapshots_set_ = true;
    // LOG_INFO("Updating snapshots @{}:", __LINE__);
    // LOG_INFO("last_child_Qbeta_snapshot_ = {}",
    //          fmt::streamed(last_child_Qbeta_snapshot_.transpose()));
    // LOG_INFO("last_child_W_snapshot_ = {}", fmt::streamed(last_child_W_snapshot_.transpose()));
  }
  edge->child_Qbeta_snapshot = child_stats.Qbeta;
  edge->child_W_snapshot = child_stats.W;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::init_edge_from_child(const GeneralContext& general_context,
                                                           Node* parent, Edge* edge) {
  const LookupTable& lookup_table = general_context.lookup_table;
  const Node* child = lookup_table.get_node(edge->child_index);
  if (!child) return;

  if (&parent->mutex() == &child->mutex()) {
    // parent and child are sharing the same mutex (this happens because mutexes are allocated from
    // a pool).
    mit::unique_lock lock(parent->mutex());
    const NodeStats& child_stats = child->stats();  // does not acquire child mutex
    edge->child_Qbeta_snapshot = child_stats.Qbeta;
    edge->child_W_snapshot = child_stats.W;
  } else {
    NodeStats child_stats = child->stats_safe();  // copy
    mit::unique_lock lock(parent->mutex());
    edge->child_Qbeta_snapshot = child_stats.Qbeta;
    edge->child_W_snapshot = child_stats.W;
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::init_node_stats_from_terminal(Node* node) {
  Base::init_node_stats_from_terminal(node);

  NodeStats& stats = node->stats();
  stats.Qbeta = stats.Q;
  stats.Qbeta_min = stats.Q;
  stats.Qbeta_max = stats.Q;
  stats.W.fill(0.f);
  stats.W_max.fill(0.f);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::undo_virtual_update(Node* node, Edge* edge) {
  throw util::CleanException("virtual updates not yet support in beta0");
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::load_evaluations(SearchContext& context) {
  Base::load_evaluations(context);

  const LookupTable& lookup_table = context.general_context->lookup_table;
  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());

    auto& stable_data = node->stable_data();
    auto& stats = node->stats();
    auto eval = item.eval();

    int n = stable_data.num_valid_actions;

    ValueArray U;
    LocalActionValueArray AU(n, kNumPlayers);

    // assumes that heads[3:4] are [value-uncertainty, action-value-uncertainty]
    //
    // TODO: we should be able to verify this assumption at compile-time
    std::copy_n(eval->data(3), U.size(), U.data());
    std::copy_n(eval->data(4), AU.size(), AU.data());

    stable_data.U = U;

    stats.Qbeta = stats.Q;
    stats.Qbeta_min = stats.Q;
    stats.Qbeta_max = stats.Q;
    stats.W = U;
    stats.W_max = U;

    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->policy_posterior_prob = edge->policy_prior_prob;
      edge->child_Qbeta_snapshot = edge->child_AV;
      edge->child_W_snapshot = AU.row(i);
    }
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::to_results(const GeneralContext& general_context,
                                                 SearchResults& results) {
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;

  const Node* root = lookup_table.get_node(root_info.node_index);
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats();  // thread-safe since single-threaded here

  core::action_mode_t mode = root->action_mode();
  group::element_t sym = root_info.canonical_sym;
  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);

  results.valid_actions.reset();
  results.policy_prior.setZero();
  results.policy_posterior.setZero();
  results.action_value_uncertainties.setZero();

  core::action_t actions[stable_data.num_valid_actions];

  int i = 0;
  for (core::action_t action : stable_data.valid_action_mask.on_indices()) {
    Game::Symmetries::apply(action, inv_sym, mode);
    results.valid_actions.set(action, true);
    actions[i] = action;

    auto* edge = lookup_table.get_edge(root, i);
    results.policy_prior(action) = edge->policy_prior_prob;
    results.policy_posterior(action) = edge->policy_posterior_prob;

    i++;
  }

  for (i = 0; i < root->stable_data().num_valid_actions; i++) {
    const Edge* edge = lookup_table.get_edge(root, i);
    const Node* child = lookup_table.get_node(edge->child_index);
    if (!child) continue;

    core::action_t action = edge->action;
    results.action_value_uncertainties.chip(action, 0) =
      eigen_util::reinterpret_as_tensor(child->stable_data().U);
  }

  Derived::load_action_symmetries(general_context, root, &actions[0], results);
  Derived::write_results(general_context, root, inv_sym, results);
  results.policy_target = results.policy_posterior;
  results.provably_lost = stats.provably_losing[stable_data.active_seat];
  results.trivial = root->trivial();

  // No policy target pruning in BetaZero

  Game::Symmetries::apply(results.counts, inv_sym, mode);
  Game::Symmetries::apply(results.Q, inv_sym, mode);
  Game::Symmetries::apply(results.Q_sq, inv_sym, mode);
  Game::Symmetries::apply(results.action_values, inv_sym, mode);

  results.win_rates = stats.Qbeta;
  results.value_prior = stable_data.R;
  results.action_mode = mode;

  results.min_win_rates = stats.Qbeta_min;
  results.max_win_rates = stats.Qbeta_max;
  results.max_uncertainties = stats.W_max;

  check_values(results.policy_target, __LINE__);
  check_values(results.min_win_rates, __LINE__);
  check_values(results.max_win_rates, __LINE__);
  check_values(results.max_uncertainties, __LINE__);
  check_values(results.action_value_uncertainties, __LINE__);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::write_to_training_info(const TrainingInfoParams& params,
                                                             TrainingInfo& training_info) {
  Base::write_to_training_info(params, training_info);

  const SearchResults* mcts_results = params.mcts_results;

  for (int p = 0; p < kNumPlayers; ++p) {
    training_info.Q_posterior(p) = mcts_results->win_rates[p];
    training_info.Q_min(p) = mcts_results->min_win_rates[p];
    training_info.Q_max(p) = mcts_results->max_win_rates[p];
    training_info.W_max(p) = mcts_results->max_uncertainties[p];
  }

  check_values(training_info.policy_target, __LINE__);
  check_values(training_info.action_values_target, __LINE__);
  check_values(training_info.Q_min, __LINE__);
  check_values(training_info.Q_max, __LINE__);
  check_values(training_info.W_max, __LINE__);

  if (params.use_for_training) {
    training_info.action_value_uncertainties_target = mcts_results->action_value_uncertainties;
    training_info.action_value_uncertainties_target_valid = true;
    check_values(training_info.action_value_uncertainties_target, __LINE__);
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::to_record(const TrainingInfo& training_info,
                                                GameLogFullRecord& full_record) {
  Base::to_record(training_info, full_record);

  full_record.Q_posterior = training_info.Q_posterior;
  full_record.Q_min = training_info.Q_min;
  full_record.Q_max = training_info.Q_max;
  full_record.W_max = training_info.W_max;

  check_values(full_record.policy_target, __LINE__);
  check_values(full_record.action_values, __LINE__);
  check_values(full_record.Q_min, __LINE__);
  check_values(full_record.Q_max, __LINE__);
  check_values(full_record.W_max, __LINE__);

  if (training_info.action_value_uncertainties_target_valid) {
    full_record.action_value_uncertainties = training_info.action_value_uncertainties_target;
  } else {
    full_record.action_value_uncertainties.setZero();
  }
  full_record.action_value_uncertainties_valid =
    training_info.action_value_uncertainties_target_valid;
  check_values(full_record.action_value_uncertainties, __LINE__);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::serialize_record(const GameLogFullRecord& full_record,
                                                       std::vector<char>& buf) {
  GameLogCompactRecord compact_record;
  compact_record.position = full_record.position;
  compact_record.Q_posterior = full_record.Q_posterior;
  compact_record.Q_min = full_record.Q_min;
  compact_record.Q_max = full_record.Q_max;
  compact_record.W_max = full_record.W_max;
  compact_record.active_seat = full_record.active_seat;
  compact_record.action_mode = Game::Rules::get_action_mode(full_record.position);
  compact_record.action = full_record.action;

  check_values(compact_record.Q_min, __LINE__);
  check_values(compact_record.Q_max, __LINE__);
  check_values(compact_record.W_max, __LINE__);
  check_values(full_record.policy_target, __LINE__);
  check_values(full_record.action_values, __LINE__);
  check_values(full_record.action_value_uncertainties, __LINE__);

  PolicyTensorData policy(full_record.policy_target_valid, full_record.policy_target);
  ActionValueTensorData action_values(full_record.action_values_valid, full_record.action_values);
  ActionValueTensorData action_value_uncertainties(full_record.action_value_uncertainties_valid,
                                                   full_record.action_value_uncertainties);

  search::GameLogCommon::write_section(buf, &compact_record, 1, false);
  policy.write_to(buf);
  action_values.write_to(buf);
  action_value_uncertainties.write_to(buf);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::to_view(const GameLogViewParams& params, GameLogView& view) {
  const GameLogCompactRecord* record = params.record;
  const GameLogCompactRecord* next_record = params.next_record;
  const State* cur_pos = params.cur_pos;
  const State* final_pos = params.final_pos;
  const GameResultTensor* outcome = params.outcome;
  group::element_t sym = params.sym;

  core::seat_index_t active_seat = record->active_seat;
  core::action_mode_t mode = record->action_mode;

  const char* addr = reinterpret_cast<const char*>(record);

  const char* policy_data_addr = addr + sizeof(GameLogCompactRecord);
  const PolicyTensorData* policy_data = reinterpret_cast<const PolicyTensorData*>(policy_data_addr);

  const char* action_values_data_addr = policy_data_addr + policy_data->size();
  const ActionValueTensorData* action_values_data =
    reinterpret_cast<const ActionValueTensorData*>(action_values_data_addr);

  const char* action_values_uncertainty_data_addr =
    action_values_data_addr + action_values_data->size();
  const ActionValueTensorData* action_values_uncertainty_data =
    reinterpret_cast<const ActionValueTensorData*>(action_values_uncertainty_data_addr);

  view.policy_valid = policy_data->load(view.policy);
  view.action_values_valid = action_values_data->load(view.action_values);
  view.action_value_uncertainties_valid =
    action_values_uncertainty_data->load(view.action_value_uncertainties);

  if (view.policy_valid) {
    Game::Symmetries::apply(view.policy, sym, mode);
  }

  if (view.action_values_valid) {
    Game::Symmetries::apply(view.action_values, sym, mode);
  }

  if (view.action_value_uncertainties_valid) {
    Game::Symmetries::apply(view.action_value_uncertainties, sym, mode);
  }

  view.next_policy_valid = false;
  if (next_record) {
    const char* next_addr = reinterpret_cast<const char*>(next_record);

    const char* next_policy_data_addr = next_addr + sizeof(GameLogCompactRecord);
    const PolicyTensorData* next_policy_data =
      reinterpret_cast<const PolicyTensorData*>(next_policy_data_addr);

    view.next_policy_valid = next_policy_data->load(view.next_policy);
    if (view.next_policy_valid) {
      Game::Symmetries::apply(view.next_policy, sym, next_record->action_mode);
    }
  }

  view.cur_pos = *cur_pos;
  view.final_pos = *final_pos;
  view.game_result = *outcome;
  view.active_seat = active_seat;
  view.Q_posterior = record->Q_posterior;
  view.Q_min = record->Q_min;
  view.Q_max = record->Q_max;
  view.W_max = record->W_max;

  check_values(view.policy, __LINE__);
  check_values(view.action_values, __LINE__);
  check_values(view.action_value_uncertainties, __LINE__);
  check_values(view.Q_min, __LINE__);
  check_values(view.Q_max, __LINE__);
  check_values(view.W_max, __LINE__);
  check_values(view.action_values, __LINE__);
  check_values(view.action_value_uncertainties, __LINE__);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_stats(NodeStats& stats, LocalPolicyArray& pi_arr,
                                                   const ValueArray& last_child_Qbeta_snapshot,
                                                   const ValueArray& last_child_W_snapshot,
                                                   const Node* node, const Edge* edge,
                                                   LookupTable& lookup_table) {
  Base::update_stats(stats, node, lookup_table);  // writes stats.Q and stats.Q_sq

  const auto& stable_data = node->stable_data();
  core::seat_index_t seat = stable_data.active_seat;
  int num_valid_actions = stable_data.num_valid_actions;

  if (stable_data.is_chance_node) {
    throw util::Exception("chance nodes not yet supported in beta0");
  } else {
    // read child stats and pi values into arrays to avoid repeated locking
    LocalActionValueArray child_Qbeta_arr(num_valid_actions, kNumPlayers);
    LocalActionValueArray child_W_arr(num_valid_actions, kNumPlayers);

    int updated_edge_arr_index = -1;

    for (int i = 0; i < num_valid_actions; i++) {
      const Edge* child_edge = lookup_table.get_edge(node, i);  // TODO: make a safe copy
      const Node* child = lookup_table.get_node(child_edge->child_index);
      if (!child) {
        child_Qbeta_arr.row(i) = child_edge->child_Qbeta_snapshot;
        child_W_arr.row(i) = child_edge->child_W_snapshot;
      } else {
        const auto child_stats = child->stats_safe();  // make a copy

        child_Qbeta_arr.row(i) = child_stats.Qbeta;
        child_W_arr.row(i) = child_stats.W;

        if (child_edge == edge) {
          updated_edge_arr_index = i;
        }
      }
    }

    RELEASE_ASSERT(updated_edge_arr_index >= 0);

    // compute posterior policy
    Derived::update_policy(pi_arr, node, edge, lookup_table, updated_edge_arr_index,
                           last_child_Qbeta_snapshot[seat], last_child_W_snapshot[seat],
                           child_Qbeta_arr.col(seat), child_W_arr.col(seat));

    // renormalize pi_arr
    float pi_sum = pi_arr.sum();
    if (pi_sum > 0.f) {
      pi_arr *= 1.0 / pi_sum;
    } else {
      pi_arr.fill(1.0f / num_valid_actions);
    }

    check_values(pi_arr, __LINE__);

    ValueArray piQbeta_sum = (child_Qbeta_arr.matrix().transpose() * pi_arr.matrix()).array();
    ValueArray piQbeta_sq_sum =
      ((child_Qbeta_arr * child_Qbeta_arr).matrix().transpose() * pi_arr.matrix()).array();
    ValueArray piW_sum = (child_W_arr.matrix().transpose() * pi_arr.matrix()).array();

    RELEASE_ASSERT(stable_data.R_valid);
    ValueArray V = Game::GameResults::to_value_array(stable_data.R);
    const ValueArray& U = stable_data.U;
    int N = stats.RN;
    RELEASE_ASSERT(N > 0);

    // Qbeta is computed as a weighted average of V (the prior value) and piQbeta_sum (the expected
    // value from children), with weights proportional to N * U (the prior-uncertainty scaled by N)
    // and piW_sum (the expected uncertainty from children). This balances the influence of the
    // prior and evidence obtained from children, gradually decaying the influence of the prior as N
    // increases.
    //
    // Should something similar be done for W? Currently, W is just the expected uncertainty from
    // children, and the prior uncertainty U is not directly incorporated into W. We're not ignoring
    // U entirely, since U influences the weight given to V in the Qbeta calculation. But it's being
    // ignored for the uncertainty calculation itself.

    ValueArray denom = piW_sum + U * N;
    ValueArray Qbeta = (V * piW_sum + U * N * piQbeta_sum) / denom;
    ValueArray W = piW_sum + piQbeta_sq_sum - piQbeta_sum * piQbeta_sum;
    // ValueArray rawW = W;
    W = W.cwiseMax(0.f);  // numerical stability

    // fix-up for imprecision when piW_sum is zero
    for (int p = 0; p < kNumPlayers; ++p) {
      if (piW_sum[p] == 0) {
        W[p] = 0;
      }
    }

    // LOG_INFO("Qbeta update:");
    // LOG_INFO("  N: {}", N);
    // LOG_INFO("  V: {}", fmt::streamed(V.transpose()));
    // LOG_INFO("  U: {}", fmt::streamed(U.transpose()));
    // LOG_INFO("  pi: {}", fmt::streamed(pi_arr.transpose()));
    // LOG_INFO("  piQbeta_sum: {}", fmt::streamed(piQbeta_sum.transpose()));
    // LOG_INFO("  piQbeta_sq_sum: {}", fmt::streamed(piQbeta_sq_sum.transpose()));
    // LOG_INFO("  piW_sum: {}", fmt::streamed(piW_sum.transpose()));
    // LOG_INFO("  Qbeta: {}", fmt::streamed(Qbeta.transpose()));
    // LOG_INFO("  rawW: {}", fmt::streamed(rawW.transpose()));
    // LOG_INFO("  W: {}", fmt::streamed(W.transpose()));

    stats.Qbeta = Qbeta;
    stats.Qbeta_min = stats.Qbeta_min.min(Qbeta);
    stats.Qbeta_max = stats.Qbeta_max.max(Qbeta);
    stats.W = W;
    stats.W_max = stats.W_max.max(W);
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_policy(LocalPolicyArray& pi_arr, const Node* node,
                                                    const Edge* edge, LookupTable& lookup_table,
                                                    int updated_edge_arr_index,
                                                    float old_child_Qbeta, float old_child_W,
                                                    const LocalPolicyArray& child_Qbeta_arr,
                                                    const LocalPolicyArray& child_W_arr) {
  // Throughout this function, theta and omega_sq represent the mean and variance of the value
  // distribution in an idealized logistic-normal model of the value distribution.

  // LOG_INFO("*** DBG update_policy() ***");
  // LOG_INFO("old_child_Qbeta: {}", old_child_Qbeta);
  // LOG_INFO("old_child_W: {}", old_child_W);
  // LOG_INFO("updated-edge: {}", updated_edge_arr_index);

  // LocalPolicyArray E_arr = pi_arr;
  // LocalPolicyArray child_Q_arr = pi_arr;
  // LocalPolicyArray dbg_pi_prior = pi_arr;

  int arr_size = pi_arr.size();
  RELEASE_ASSERT(updated_edge_arr_index >= 0 && updated_edge_arr_index < arr_size);

  // auto seat = node->stable_data().active_seat;
  // for (int i = 0; i < arr_size; i++) {
  //   Edge* child_edge = lookup_table.get_edge(node, i);
  //   E_arr[i] = child_edge->E;
  //   Node* child = lookup_table.get_node(child_edge->child_index);
  //   child_Q_arr[i] = child ? child->stats_safe().Q(seat) : child_edge->child_AV[seat];
  // }

  double theta_old;
  double omega_sq_old;
  compute_theta_omega_sq(old_child_Qbeta, old_child_W, theta_old, omega_sq_old);

  // LOG_INFO("theta_old: {}", theta_old);
  // LOG_INFO("omega_sq_old: {}", omega_sq_old);

  LocalPolicyArrayDouble theta_arr(arr_size);
  LocalPolicyArrayDouble omega_sq_arr(arr_size);

  for (int i = 0; i < arr_size; i++) {
    compute_theta_omega_sq(child_Qbeta_arr[i], child_W_arr[i], theta_arr[i], omega_sq_arr[i]);
  }

  math::finiteness_t finiteness_arr[arr_size];
  bool any_pos_inf = false;
  bool any_neg_inf = false;
  for (int i = 0; i < arr_size; i++) {
    finiteness_arr[i] = math::get_finiteness(theta_arr[i]);
    any_pos_inf |= (finiteness_arr[i] == math::kPosInf);
    any_neg_inf |= (finiteness_arr[i] == math::kNegInf);
  }

  if (any_neg_inf) {
    // Zero out pi values for all -inf theta actions
    for (int i = 0; i < arr_size; i++) {
      if (finiteness_arr[i] == math::kNegInf) {
        pi_arr[i] = 0;
      }
    }
  }

  if (any_pos_inf) {
    // Collapse pi values to only those with +inf theta
    for (int i = 0; i < arr_size; i++) {
      pi_arr[i] = (finiteness_arr[i] == math::kPosInf) ? 1.0f : 0.f;
    }
    // LOG_INFO("skipping update_policy due to +inf theta");
    // std::stringstream ss;
    // auto action_data =
    //   eigen_util::concatenate_columns(dbg_pi_prior, E_arr, child_Q_arr, child_Qbeta_arr,
    //                                   child_W_arr, theta_arr, omega_sq_arr, pi_arr);
    // static std::vector<std::string> action_columns = {
    //   "pi-prior", "E", "child_Q", "child_Qbeta", "child_W", "theta", "omega_sq", "pi-posterior"};
    // eigen_util::print_array(ss, action_data, action_columns);
    // LOG_INFO("\n{}", ss.str());
    return;
  }

  double theta_new = theta_arr[updated_edge_arr_index];
  double omega_sq_new = omega_sq_arr[updated_edge_arr_index];
  math::finiteness_t finiteness_new = finiteness_arr[updated_edge_arr_index];

  if (finiteness_new == math::kNegInf) {
    // LOG_INFO("skipping update_policy due to -inf theta");
    // std::stringstream ss;
    // auto action_data =
    //   eigen_util::concatenate_columns(dbg_pi_prior, E_arr, child_Q_arr, child_Qbeta_arr,
    //                                   child_W_arr, theta_arr, omega_sq_arr, pi_arr);
    // static std::vector<std::string> action_columns = {
    //   "pi-prior", "E", "child_Q", "child_Qbeta", "child_W", "theta", "omega_sq", "pi-posterior"};
    // eigen_util::print_array(ss, action_data, action_columns);
    // LOG_INFO("\n{}", ss.str());
    return;
  }

  if (omega_sq_new > 0) {
    RELEASE_ASSERT(omega_sq_old > 0);  // cannot go from certain to uncertain
  } else {                             // zero uncertainty case
    // check for domination by another zero-uncertainty action
    for (int i = 0; i < arr_size; i++) {
      if (i == updated_edge_arr_index) continue;
      if (omega_sq_arr[i] == 0) {
        if (theta_arr[i] > theta_new) {
          // dominated
          pi_arr[updated_edge_arr_index] = 0;
          // LOG_INFO("skipping update_policy due to domination");
          // std::stringstream ss;
          // auto action_data =
          //   eigen_util::concatenate_columns(dbg_pi_prior, E_arr, child_Q_arr, child_Qbeta_arr,
          //                                   child_W_arr, theta_arr, omega_sq_arr, pi_arr);
          // static std::vector<std::string> action_columns = {"pi-prior",    "E",           "child_Q",
          //                                                   "child_Qbeta", "child_W",     "theta",
          //                                                   "omega_sq",    "pi-posterior"};
          // eigen_util::print_array(ss, action_data, action_columns);
          // LOG_INFO("\n{}", ss.str());
          return;
        } else if (theta_arr[i] == theta_new) {
          // tie
          pi_arr[updated_edge_arr_index] = pi_arr[i];
          // LOG_INFO("skipping update_policy due to tie");
          // std::stringstream ss;
          // auto action_data =
          //   eigen_util::concatenate_columns(dbg_pi_prior, E_arr, child_Q_arr, child_Qbeta_arr,
          //                                   child_W_arr, theta_arr, omega_sq_arr, pi_arr);
          // static std::vector<std::string> action_columns = {"pi-prior",    "E",           "child_Q",
          //                                                   "child_Qbeta", "child_W",     "theta",
          //                                                   "omega_sq",    "pi-posterior"};
          // eigen_util::print_array(ss, action_data, action_columns);
          // LOG_INFO("\n{}", ss.str());
          return;
        }
      }
    }
  }

  // At this point, we know that there are no +inf thetas, and the updated action is not -inf.
  // If there is zero uncertainty with the updated action, we know that it dominates all other
  // zero-uncertainty actions.
  //
  // We can now compute the posterior pi values using the logistic-normal model.

  LocalPolicyArrayDouble alpha_arr(arr_size);
  LocalPolicyArrayDouble beta_arr(arr_size);
  alpha_arr.setConstant(1);
  beta_arr.setZero();

  for (int i = 0; i < arr_size; i++) {
    bool was_updated = (i == updated_edge_arr_index);
    if (was_updated) {
      continue;
    }

    if (finiteness_arr[i] == math::kNegInf) {
      RELEASE_ASSERT(pi_arr[i] == 0.f);
      continue;
    }

    double theta_i = theta_arr[i];
    double omega_sq_i = omega_sq_arr[i];

    if (omega_sq_i == 0) {  // certain action
      if (omega_sq_new == 0) {
        RELEASE_ASSERT(theta_new > theta_i);  // expect strict domination
        pi_arr[i] = 0.f;
        continue;
      }
    }

    double alpha = detail::alpha_from_thetas(theta_old, omega_sq_old, theta_new, omega_sq_new,
                                             theta_i, omega_sq_i);
    double beta = pi_arr[i];

    alpha_arr[i] = alpha;
    beta_arr[i] = beta;
  }

  double beta_sum = beta_arr.sum();
  if (beta_sum > 0.0) {
    beta_arr *= (1.0 / beta_sum);
    LocalPolicyArrayDouble log_pi_arr = pi_arr.template cast<double>().log();
    double adjustment = (beta_arr * alpha_arr).sum();
    adjustment = std::max(-5.0, std::min(+5.0, adjustment));  // cap to reasonable range
    log_pi_arr[updated_edge_arr_index] += adjustment;
    pi_arr = log_pi_arr.template cast<float>();
    eigen_util::softmax_in_place(pi_arr);
  }

  // std::stringstream ss;
  // auto action_data =
  //   eigen_util::concatenate_columns(dbg_pi_prior, E_arr, child_Q_arr, child_Qbeta_arr, child_W_arr,
  //                                   theta_arr, omega_sq_arr, alpha_arr, beta_arr, pi_arr);
  // static std::vector<std::string> action_columns = {
  //   "pi-prior", "E",        "child_Q", "child_Qbeta", "child_W",
  //   "theta",    "omega_sq", "alpha",   "beta",        "pi-posterior"};
  // eigen_util::print_array(ss, action_data, action_columns);
  // LOG_INFO("\n{}", ss.str());
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::compute_theta_omega_sq(double Qbeta, double W, double& theta,
                                                             double& omega_sq) {
  constexpr double kMin = Game::GameResults::kMinValue;
  constexpr double kMax = Game::GameResults::kMaxValue;

  if (Qbeta <= kMin) {
    theta = -std::numeric_limits<double>::infinity();
    omega_sq = 0;
    return;
  } else if (Qbeta >= kMax) {
    theta = +std::numeric_limits<double>::infinity();
    omega_sq = 0;
    return;
  }

  double mu = Qbeta;

  // Rescale Qbeta to [0, 1]
  mu = (mu - kMin) / (kMax - kMin);
  mu = std::max(0.0, std::min(1.0, mu));

  // Rescale W to variance in [0, 1]
  double sigma_sq = W;
  sigma_sq /= (kMax - kMin) * (kMax - kMin);
  sigma_sq = std::max(0.0, std::min(1.0, sigma_sq));

  // TODO: cache theta/omega_sq values in NodeStats to avoid recomputation

  double theta1 = std::log(mu / (1 - mu));
  double theta2 = -(1 - 2 * mu) * sigma_sq / (2 * mu * mu * (1 - mu) * (1 - mu));
  theta = theta1 + theta2;

  omega_sq = sigma_sq / (mu * mu * (1 - mu) * (1 - mu));
}

}  // namespace beta0
