#include "betazero/Algorithms.hpp"

#include "core/BasicTypes.hpp"
#include "util/Asserts.hpp"
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
  const double z_old = (theta_old - theta_i) / std::sqrt(omega_sq_old + omega_sq_i);
  const double z_new = (theta_new - theta_i) / std::sqrt(omega_sq_new + omega_sq_i);

  return math::normal_cdf_logit_diff(z_new, z_old);
}

}  // namespace detail

template <search::concepts::Traits Traits, typename Derived>
template <typename MutexProtectedFunc>
void AlgorithmsBase<Traits, Derived>::Backpropagator::run(Node* node, Edge* edge,
                                                          MutexProtectedFunc&& func) {
  mit::unique_lock lock(node->mutex());
  // LOG_INFO("Backpropagator::run()");
  func();
  if (!snapshots_set_ && edge) {
    update_edge_snapshots(node, edge, true);
  }

  if (!edge) return;

  RELEASE_ASSERT(snapshots_set_);
  NodeStats stats = node->stats();  // copy

  ValueArray tmp_last_child_Q_snapshot = stats.Q;
  ValueArray tmp_last_child_W_snapshot = stats.W;

  LookupTable& lookup_table = context_.general_context->lookup_table;
  int num_valid_actions = node->stable_data().num_valid_actions;
  LocalPolicyArray pi_arr(num_valid_actions);
  for (int i = 0; i < num_valid_actions; i++) {
    const Edge* child_edge = lookup_table.get_edge(node, i);
    pi_arr(i) = child_edge->policy_posterior_prob;
  }
  lock.unlock();

  Derived::update_stats(context_, stats, pi_arr, last_child_Q_snapshot_, last_child_W_snapshot_,
                        node, edge);

  // LOG_INFO("Update result:");
  // LOG_INFO("Q = {} -> {}", fmt::streamed(tmp_last_child_Q_snapshot.transpose()),
  //          fmt::streamed(stats.Q.transpose()));
  // LOG_INFO("W = {} -> {}", fmt::streamed(tmp_last_child_W_snapshot.transpose()),
  //          fmt::streamed(stats.W.transpose()));

  lock.lock();

  update_edge_snapshots(node, edge);
  last_child_Q_snapshot_ = tmp_last_child_Q_snapshot;
  last_child_W_snapshot_ = tmp_last_child_W_snapshot;
  // LOG_INFO("Updating snapshots @{}:", __LINE__);
  // LOG_INFO("last_child_Q_snapshot_ = {}",
  //          fmt::streamed(last_child_Q_snapshot_.transpose()));
  // LOG_INFO("last_child_W_snapshot_ = {}", fmt::streamed(last_child_W_snapshot_.transpose()));
  for (int i = 0; i < num_valid_actions; i++) {
    Edge* child_edge = lookup_table.get_edge(node, i);
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

  LookupTable& lookup_table = context_.general_context->lookup_table;
  const Node* child = lookup_table.get_node(edge->child_index);
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
    last_child_Q_snapshot_ = edge->child_Q_snapshot;
    last_child_W_snapshot_ = edge->child_W_snapshot;
    snapshots_set_ = true;
    // LOG_INFO("Updating snapshots @{}:", __LINE__);
    // LOG_INFO("last_child_Q_snapshot_ = {}",
    //          fmt::streamed(last_child_Q_snapshot_.transpose()));
    // LOG_INFO("last_child_W_snapshot_ = {}", fmt::streamed(last_child_W_snapshot_.transpose()));
  }
  edge->child_Q_snapshot = child_stats.Q;
  edge->child_W_snapshot = child_stats.W;
  eigen_util::debug_validate_bounds(edge->child_Q_snapshot);
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
    edge->child_Q_snapshot = child_stats.Q;
    edge->child_W_snapshot = child_stats.W;
    eigen_util::debug_validate_bounds(edge->child_Q_snapshot);
  } else {
    NodeStats child_stats = child->stats_safe();  // copy
    mit::unique_lock lock(parent->mutex());
    edge->child_Q_snapshot = child_stats.Q;
    edge->child_W_snapshot = child_stats.W;
    eigen_util::debug_validate_bounds(edge->child_Q_snapshot);
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::init_node_stats_from_terminal(Node* node) {
  Base::init_node_stats_from_terminal(node);

  NodeStats& stats = node->stats();
  stats.Q_min = stats.Q;
  stats.Q_max = stats.Q;
  stats.W.fill(0.f);
  stats.W_max.fill(0.f);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::undo_virtual_update(Node* node, Edge* edge) {
  throw util::CleanException("virtual updates not yet support in beta0");
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::load_evaluations(SearchContext& context) {
  const LookupTable& lookup_table = context.general_context->lookup_table;
  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());

    auto& stable_data = node->stable_data();
    auto& stats = node->stats();
    auto eval = item.eval();

    int n = stable_data.num_valid_actions;

    GameResultTensor R;
    ValueArray U;
    LocalActionValueArray AU(n, kNumPlayers);
    LocalPolicyArray P_raw(n);
    LocalActionValueArray AV(n, kNumPlayers);

    // assumes that heads are in order policy, value, action-value
    //
    // TODO: we should be able to verify this assumption at compile-time
    std::copy_n(eval->data(0), P_raw.size(), P_raw.data());
    std::copy_n(eval->data(1), R.size(), R.data());
    std::copy_n(eval->data(2), AV.size(), AV.data());

    // assumes that heads[3:4] are [value-uncertainty, action-value-uncertainty]
    //
    // TODO: we should be able to verify this assumption at compile-time
    std::copy_n(eval->data(3), U.size(), U.data());
    std::copy_n(eval->data(4), AU.size(), AU.data());

    LocalPolicyArray P_adjusted = P_raw;
    Derived::transform_policy(context, P_adjusted);

    stable_data.R = R;
    stable_data.R_valid = true;

    stable_data.U = U;

    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->policy_prior_prob = P_raw[i];
      edge->adjusted_base_prob = P_adjusted[i];
      edge->child_AV = AV.row(i);
      edge->policy_posterior_prob = edge->policy_prior_prob;
      edge->child_Q_snapshot = edge->child_AV;
      edge->child_W_snapshot = AU.row(i);
      eigen_util::debug_validate_bounds(edge->child_Q_snapshot);
    }

    RELEASE_ASSERT(stats.RN == 0, "RN={}", stats.RN);

    ValueArray V = Game::GameResults::to_value_array(R);
    stats.Q = V;
    stats.Q_sq = V * V;

    LocalActionValueArray child_Q_arr(n, kNumPlayers);
    const LocalActionValueArray& child_W_arr = AU;
    LocalPolicyArray pi_arr(n);
    for (int i = 0; i < n; i++) {
      const Edge* edge = lookup_table.get_edge(node, i);
      pi_arr(i) = edge->policy_posterior_prob;
      child_Q_arr.row(i) = edge->child_Q_snapshot;
    }

    update_QW_fields(context, node, stable_data, pi_arr, child_Q_arr, child_W_arr, stats);

    const RootInfo& root_info = context.general_context->root_info;
    Node* root = lookup_table.get_node(root_info.node_index);
    if (root) {
      root->stats().RN = std::max(root->stats().RN, 1);
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
  Game::Symmetries::apply(results.action_value_uncertainties, inv_sym, mode);

  results.win_rates = stats.Q;
  results.value_prior = stable_data.R;
  results.action_mode = mode;

  results.min_win_rates = stats.Q_min;
  results.max_win_rates = stats.Q_max;
  results.max_uncertainties = stats.W_max;

  eigen_util::assert_is_valid_prob_distr(results.policy_posterior);
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

  if (params.use_for_training) {
    training_info.action_value_uncertainties_target = mcts_results->action_value_uncertainties;
    training_info.action_value_uncertainties_target_valid = true;
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

  if (training_info.action_value_uncertainties_target_valid) {
    full_record.action_value_uncertainties = training_info.action_value_uncertainties_target;
  } else {
    full_record.action_value_uncertainties.setZero();
  }
  full_record.action_value_uncertainties_valid =
    training_info.action_value_uncertainties_target_valid;
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
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_stats(SearchContext& context, NodeStats& stats,
                                                   LocalPolicyArray& pi_arr,
                                                   const ValueArray& last_child_Q_snapshot,
                                                   const ValueArray& last_child_W_snapshot,
                                                   const Node* node, const Edge* edge) {
  LookupTable& lookup_table = context.general_context->lookup_table;
  const auto& stable_data = node->stable_data();
  core::seat_index_t seat = stable_data.active_seat;
  int num_valid_actions = stable_data.num_valid_actions;

  player_bitset_t all_provably_winning;
  player_bitset_t all_provably_losing;
  all_provably_winning.set();
  all_provably_losing.set();

  if (stable_data.is_chance_node) {
    throw util::Exception("chance nodes not yet supported in beta0");
  } else {
    // read child stats and pi values into arrays to avoid repeated locking
    LocalActionValueArray child_Q_arr(num_valid_actions, kNumPlayers);
    LocalActionValueArray child_W_arr(num_valid_actions, kNumPlayers);

    int updated_edge_arr_index = -1;
    bool cp_has_winning_move = false;
    int num_expanded_edges = 0;

    for (int i = 0; i < num_valid_actions; i++) {
      const Edge* child_edge = lookup_table.get_edge(node, i);  // TODO: make a safe copy
      const Node* child = lookup_table.get_node(child_edge->child_index);
      if (!child) {
        child_Q_arr.row(i) = child_edge->child_Q_snapshot;
        child_W_arr.row(i) = child_edge->child_W_snapshot;
      } else {
        const auto child_stats = child->stats_safe();  // make a copy

        child_Q_arr.row(i) = child_stats.Q;
        child_W_arr.row(i) = child_stats.W;

        eigen_util::debug_validate_bounds(child_stats.Q);
        if (child_edge == edge) {
          updated_edge_arr_index = i;
        }

        cp_has_winning_move |= child_stats.provably_winning[seat];
        all_provably_winning &= child_stats.provably_winning;
        all_provably_losing &= child_stats.provably_losing;
        num_expanded_edges++;
      }
    }

    eigen_util::debug_validate_bounds(child_Q_arr);
    bool all_edges_expanded = (num_expanded_edges == num_valid_actions);
    if (!all_edges_expanded) {
      all_provably_winning.reset();
      all_provably_losing.reset();
    }

    RELEASE_ASSERT(updated_edge_arr_index >= 0);

    // compute posterior policy
    Derived::update_policy(context, pi_arr, node, edge, lookup_table, updated_edge_arr_index,
                           last_child_Q_snapshot[seat], last_child_W_snapshot[seat],
                           child_Q_arr.col(seat), child_W_arr.col(seat));

    // renormalize pi_arr
    float pi_sum = pi_arr.sum();
    if (pi_sum > 0.f) {
      LocalPolicyArray posterior = pi_arr;
      posterior *= 1.0 / pi_sum;
      if (posterior.hasNaN()) {
        for (int i = 0; i < pi_arr.size(); ++i) {
          posterior[i] = pi_arr[i] > 0.f ? 1.f : 0.f;
        }
        posterior *= 1.0 / posterior.sum();
      }
      pi_arr = posterior;
    } else {
      RELEASE_ASSERT(num_valid_actions > 0);
      pi_arr.fill(1.0f / num_valid_actions);
    }

    update_QW_fields(context, node, stable_data, pi_arr, child_Q_arr, child_W_arr, stats);
    if (cp_has_winning_move) {
      stats.provably_winning[seat] = true;
      stats.provably_losing.set();
      stats.provably_losing[seat] = false;
    } else if (all_edges_expanded) {
      stats.provably_winning = all_provably_winning;
      stats.provably_losing = all_provably_losing;
    }
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_policy(
  SearchContext& context, LocalPolicyArray& pi_arr, const Node* node, const Edge* edge,
  LookupTable& lookup_table, int updated_edge_arr_index, float old_child_Q, float old_child_W,
  const LocalPolicyArray& child_Q_arr, const LocalPolicyArray& child_W_arr) {
  // Throughout this function, theta and omega_sq represent the mean and variance of the value
  // distribution in an idealized logistic-normal model of the value distribution.

  LocalPolicyArray dbg_pi_prior = pi_arr;

  int arr_size = pi_arr.size();
  RELEASE_ASSERT(updated_edge_arr_index >= 0 && updated_edge_arr_index < arr_size);

  double theta_old;
  double omega_sq_old;
  compute_theta_omega_sq(old_child_Q, old_child_W, theta_old, omega_sq_old);

  LocalPolicyArrayDouble theta_arr(arr_size);
  LocalPolicyArrayDouble omega_sq_arr(arr_size);

  for (int i = 0; i < arr_size; i++) {
    compute_theta_omega_sq(child_Q_arr[i], child_W_arr[i], theta_arr[i], omega_sq_arr[i]);
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

  LocalPolicyArrayDouble alpha_arr(arr_size);
  LocalPolicyArrayDouble beta_arr(arr_size);
  alpha_arr.setConstant(1);
  beta_arr.setZero();

  auto debug_dump = [&](const std::string& msg) {
    if (!search::kEnableSearchDebug) return;
    LocalPolicyArray E_arr = pi_arr;
    group::element_t inv_sym = Game::SymmetryGroup::inverse(context.leaf_canonical_sym);

    LocalPolicyArray actions = pi_arr;
    for (int i = 0; i < arr_size; i++) {
      Edge* child_edge = lookup_table.get_edge(node, i);
      core::action_t action = child_edge->action;
      Game::Symmetries::apply(action, inv_sym, node->action_mode());
      actions(i) = action;
      E_arr[i] = child_edge->E;
    }

    LOG_INFO("*** DBG update_policy() ***");
    LOG_INFO("old_child_Q: {}", old_child_Q);
    LOG_INFO("old_child_W: {}", old_child_W);
    LOG_INFO("updated-edge: {}", updated_edge_arr_index);
    LOG_INFO("theta_old: {}", theta_old);
    LOG_INFO("omega_sq_old: {}", omega_sq_old);

    std::stringstream ss;
    auto action_data = eigen_util::sort_rows(
      eigen_util::concatenate_columns(actions, dbg_pi_prior, E_arr, child_Q_arr, child_W_arr,
                                      theta_arr, omega_sq_arr, alpha_arr, beta_arr, pi_arr));

    std::vector<std::string> action_columns = {"action", "pi-prior", "E",     "child_Q", "child_W",
                                               "theta",  "omega_sq", "alpha", "beta",    "pi"};

    eigen_util::PrintArrayFormatMap fmt_map{
      {"action", [&](float x) { return Game::IO::action_to_str(x, node->action_mode()); }},
    };

    eigen_util::print_array(ss, action_data, action_columns, &fmt_map);
    LOG_INFO("{}\n{}", msg, ss.str());
  };

  if (any_pos_inf) {
    // Collapse pi values to only those with +inf theta
    for (int i = 0; i < arr_size; i++) {
      pi_arr[i] = (finiteness_arr[i] == math::kPosInf) ? 1.0f : 0.f;
    }
    debug_dump("skipping due to +inf theta");
    return;
  }

  double theta_new = theta_arr[updated_edge_arr_index];
  double omega_sq_new = omega_sq_arr[updated_edge_arr_index];
  math::finiteness_t finiteness_new = finiteness_arr[updated_edge_arr_index];

  if (finiteness_new == math::kNegInf) {
    debug_dump("skipping due to -inf theta");
    return;
  }

  if (omega_sq_new <= 0) {
    // check for domination by another zero-uncertainty action
    for (int i = 0; i < arr_size; i++) {
      if (i == updated_edge_arr_index) continue;
      if (omega_sq_arr[i] == 0) {
        if (theta_arr[i] > theta_new) {
          // dominated
          pi_arr[updated_edge_arr_index] = 0;
          debug_dump("skipping due to domination");
          return;
        } else if (theta_arr[i] == theta_new) {
          // tie
          pi_arr[updated_edge_arr_index] = pi_arr[i];
          debug_dump("skipping update_policy due to tie");
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
    adjustment = std::max(-1.0, std::min(+1.0, adjustment));  // cap to reasonable range
    log_pi_arr[updated_edge_arr_index] += adjustment;
    pi_arr = log_pi_arr.template cast<float>();
    eigen_util::softmax_in_place(pi_arr);
  }

  debug_dump("after update_policy");
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::compute_theta_omega_sq(double Q, double W, double& theta,
                                                             double& omega_sq) {
  constexpr double kMin = Game::GameResults::kMinValue;
  constexpr double kMax = Game::GameResults::kMaxValue;

  if (Q <= kMin) {
    theta = -std::numeric_limits<double>::infinity();
    omega_sq = 0;
    return;
  } else if (Q >= kMax) {
    theta = +std::numeric_limits<double>::infinity();
    omega_sq = 0;
    return;
  }

  double mu = Q;

  // Rescale Q to [0, 1]
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

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_QW_fields(SearchContext& context, const Node* node,
                                                       const NodeStableData& stable_data,
                                                       const LocalPolicyArray& pi_arr,
                                                       const LocalActionValueArray& child_Q_arr,
                                                       const LocalActionValueArray& child_W_arr,
                                                       NodeStats& stats) {
  RELEASE_ASSERT(stable_data.R_valid);

  ValueArray V = Game::GameResults::to_value_array(stable_data.R);
  const ValueArray& U = stable_data.U;

  ValueArray old_Q = stats.Q;
  ValueArray old_W = stats.W;

  if (stats.RN == 0) {
    stats.Q = V;
    stats.W = U;
    eigen_util::debug_validate_bounds(stats.Q);
  } else {
    // RELEASE_ASSERT(stats.RN >= 2, "RN={}", stats.RN);
    int N = std::max(stats.RN, 2);

    // We have length-n vector pi, Q, and W for the children.
    //
    // To incorporate the prior (V, U) with the children's statistics, we extend to a length-(n+1)
    // vector by adding the prior as an additional "virtual child" with mean V and uncertainty U
    // and a carefully chosen weight alpha.
    //
    // This choice of alpha ensures that the effective sample size of the virtual child decays
    // like 1/N, and that it gets zeroed out if the policy is fully concentrated on children with
    // zero uncertainty.
    //
    // The other children's weights are scaled by beta = (1 - alpha).

    ValueArray piQ_sum = (child_Q_arr.matrix().transpose() * pi_arr.matrix()).array();
    ValueArray piW_sum = (child_W_arr.matrix().transpose() * pi_arr.matrix()).array();

    float alpha = (2 * piW_sum.sum()) / (N * (U.sum() + piW_sum.sum()));
    float beta = 1.0f - alpha;
    RELEASE_ASSERT(alpha >= 0.f && alpha <= 1.f);

    ValueArray Q_bar = alpha * V + beta * piQ_sum;
    auto Qdiff = child_Q_arr.rowwise() - Q_bar.transpose();
    ValueArray piQdiff2 = (Qdiff.square().matrix().transpose() * pi_arr.matrix()).array();
    ValueArray W = alpha * (U + (V - Q_bar).square()) + beta * (piW_sum + piQdiff2);

    if (!Q_bar.isFinite().all()) {
      LOG_ERROR("Q_bar has non-finite values:");
      LOG_ERROR("N = {}", N);
      LOG_ERROR("V = {}", fmt::streamed(V.transpose()));
      LOG_ERROR("U = {}", fmt::streamed(U.transpose()));
      LOG_ERROR("pi_arr = {}", fmt::streamed(pi_arr.transpose()));
      LOG_ERROR("child_Q_arr =\n{}", fmt::streamed(child_Q_arr));
      LOG_ERROR("child_W_arr =\n{}", fmt::streamed(child_W_arr));
      LOG_ERROR("alpha = {}", alpha);
      LOG_ERROR("beta = {}", beta);
      LOG_ERROR("piQ_sum = {}", fmt::streamed(piQ_sum.transpose()));
      LOG_ERROR("Q_bar = {}", fmt::streamed(Q_bar.transpose()));
      RELEASE_ASSERT(false);
    }

    stats.Q = Q_bar;
    stats.W = W;
    eigen_util::debug_validate_bounds(stats.Q);
  }

  stats.Q_min = stats.Q_min.cwiseMin(stats.Q);
  stats.Q_max = stats.Q_max.cwiseMax(stats.Q);
  stats.W_max = stats.W_max.cwiseMax(stats.W);

  if (search::kEnableSearchDebug) {
    std::ostringstream ss;
    ss << std::format("{:>{}}", "", context.log_prefix_n());

    std::string line_break =
      std::format("\n{:>{}}", "", util::Logging::kTimestampPrefixLength + context.log_prefix_n());

    ss << "QW update details:\n";
    ss << "  RN: " << stats.RN << line_break;

    core::seat_index_t seat = stable_data.active_seat;

    ValueArray players;
    ValueArray CP;
    for (int p = 0; p < kNumPlayers; ++p) {
      players(p) = p;
      CP(p) = p == seat;
    }

    static std::vector<std::string> player_columns = {"Seat", "V",    "U",    "oldQ",
                                                      "oldW", "newQ", "newW", "CurP"};

    auto player_data =
      eigen_util::concatenate_columns(players, V, U, old_Q, old_W, stats.Q, stats.W, CP);

    eigen_util::PrintArrayFormatMap fmt_map1{
      {"Seat", [&](float x) { return std::to_string(int(x)); }},
      {"CurP", [&](float x) { return std::string(x ? "*" : ""); }},
    };

    std::stringstream ss1;
    eigen_util::print_array(ss1, player_data, player_columns, &fmt_map1);

    for (const std::string& line : util::splitlines(ss1.str())) {
      ss << line << line_break;
    }

    LookupTable& lookup_table = context.general_context->lookup_table;
    int n_actions = stable_data.num_valid_actions;
    group::element_t inv_sym = Game::SymmetryGroup::inverse(context.leaf_canonical_sym);
    LocalPolicyArray actions(n_actions);
    LocalPolicyArray cQ_arr(n_actions);
    LocalPolicyArray cW_arr(n_actions);
    for (int e = 0; e < n_actions; ++e) {
      auto edge = lookup_table.get_edge(node, e);
      core::action_t action = edge->action;
      Game::Symmetries::apply(action, inv_sym, stable_data.action_mode);
      actions(e) = action;
      cQ_arr(e) = child_Q_arr(e, seat);
      cW_arr(e) = child_W_arr(e, seat);
    }

    static std::vector<std::string> action_columns = {"action", "child_Q_arr", "child_W_arr", "pi"};

    auto action_data =
      eigen_util::sort_rows(eigen_util::concatenate_columns(actions, cQ_arr, cW_arr, pi_arr));

    eigen_util::PrintArrayFormatMap fmt_map2{
      {"action", [&](float x) { return Game::IO::action_to_str(x, node->action_mode()); }},
    };

    std::stringstream ss2;
    eigen_util::print_array(ss2, action_data, action_columns, &fmt_map2);

    for (const std::string& line : util::splitlines(ss2.str())) {
      ss << line << line_break;
    }
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::print_action_selection_details(const SearchContext& context,
                                                                     const PuctCalculator& selector,
                                                                     int argmax_index) {
  LookupTable& lookup_table = context.general_context->lookup_table;
  Node* node = context.visit_node;
  if (search::kEnableSearchDebug) {
    std::ostringstream ss;
    ss << std::format("{:>{}}", "", context.log_prefix_n());

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

    eigen_util::PrintArrayFormatMap fmt_map1{
      {"Seat", [&](float x) { return std::to_string(int(x)); }},
      {"CurP", [&](float x) { return std::string(x ? "*" : ""); }},
    };

    std::stringstream ss1;
    eigen_util::print_array(ss1, player_data, player_columns, &fmt_map1);

    std::string line_break =
      std::format("\n{:>{}}", "", util::Logging::kTimestampPrefixLength + context.log_prefix_n());

    for (const std::string& line : util::splitlines(ss1.str())) {
      ss << line << line_break;
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
    LocalPolicyArray pi(n_actions);
    LocalPolicyArray W(n_actions);
    child_addr.setConstant(-1);
    argmax.setZero();
    argmax(argmax_index) = 1;

    group::element_t inv_sym = Game::SymmetryGroup::inverse(context.leaf_canonical_sym);
    for (int e = 0; e < n_actions; ++e) {
      auto edge = lookup_table.get_edge(node, e);
      core::action_t action = edge->action;
      Game::Symmetries::apply(action, inv_sym, node->action_mode());
      actions(e) = action;
      child_addr(e) = edge->child_index;
      pi(e) = edge->policy_posterior_prob;
      W(e) = edge->child_W_snapshot[seat];
    }

    static std::vector<std::string> action_columns = {"action", "pi", "W",   "P",    "Q",
                                                      "FPU",    "PW", "PL",  "E",    "mE",
                                                      "RN",     "VN", "&ch", "PUCT", "argmax"};
    auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
      actions, pi, W, P, Q, FPU, PW, PL, E, mE, RN, VN, child_addr, PUCT, argmax));

    eigen_util::PrintArrayFormatMap fmt_map2{
      {"action", [&](float x) { return Game::IO::action_to_str(x, node->action_mode()); }},
      {"&ch", [](float x) { return x < 0 ? std::string() : std::to_string((int)x); }},
      {"argmax", [](float x) { return std::string(x == 0 ? "" : "*"); }},
    };

    std::stringstream ss2;
    eigen_util::print_array(ss2, action_data, action_columns, &fmt_map2);

    for (const std::string& line : util::splitlines(ss2.str())) {
      ss << line << line_break;
    }

    LOG_INFO(ss.str());
  }
}

}  // namespace beta0
