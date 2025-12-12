#include "betazero/Algorithms.hpp"

#include "core/BasicTypes.hpp"
#include "util/Asserts.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/Gaussian1D.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <cmath>

namespace beta0 {

template <search::concepts::Traits Traits, typename Derived>
template <typename MutexProtectedFunc>
void AlgorithmsBase<Traits, Derived>::Backpropagator::run(Node* node, Edge* edge,
                                                          MutexProtectedFunc&& func) {
  mit::unique_lock lock(node->mutex());
  func();

  if (!edge) return;

  NodeStats stats = node->stats();  // copy

  LookupTable& lookup_table = context_.general_context->lookup_table;
  int num_valid_actions = node->stable_data().num_valid_actions;
  LocalPolicyArray pi_arr(num_valid_actions);
  for (int i = 0; i < num_valid_actions; i++) {
    const Edge* child_edge = lookup_table.get_edge(node, i);
    pi_arr(i) = child_edge->policy_posterior_prob;
  }
  lock.unlock();

  Derived::update_stats(context_, stats, pi_arr, node, edge);

  lock.lock();
  for (int i = 0; i < num_valid_actions; i++) {
    Edge* child_edge = lookup_table.get_edge(node, i);
    child_edge->policy_posterior_prob = pi_arr[i];
  }

  node->stats() = stats;  // copy back
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::init_node_stats_from_terminal(Node* node) {
  Base::init_node_stats_from_terminal(node);

  NodeStats& stats = node->stats();
  populate_logit_value_beliefs(stats.Q, stats.W, stats.logit_value_beliefs);
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
int AlgorithmsBase<Traits, Derived>::get_best_child_index(const SearchContext& context) {
  throw util::CleanException("get_best_child_index not yet support in beta0");
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

    // TODO: perform a massaging step here so that (P_raw, AU, AV) are consistent with R and U.

    LocalPolicyArray P_adjusted = P_raw;
    Derived::transform_policy(context, P_adjusted);

    ValueArray V = Game::GameResults::to_value_array(R);

    stable_data.R = R;
    stable_data.R_valid = true;
    stable_data.U = U;
    populate_logit_value_beliefs(V, U, stable_data.prior_logit_value_beliefs);

    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->policy_prior_prob = P_raw[i];
      edge->adjusted_base_prob = P_adjusted[i];
      edge->child_AU = AU.row(i);
      edge->child_AV = AV.row(i);
      edge->policy_posterior_prob = edge->policy_prior_prob;
      populate_logit_value_beliefs(edge->child_AV, edge->child_AU, edge->child_logit_value_beliefs);
    }

    RELEASE_ASSERT(stats.RN == 0, "RN={}", stats.RN);

    stats.Q = V;
    stats.Q_sq = V * V;
    stats.Q_min = stats.Q_min.cwiseMin(stats.Q);
    stats.Q_max = stats.Q_max.cwiseMax(stats.Q);
    stats.W_max = stats.W_max.cwiseMax(stats.W);
  }

  const RootInfo& root_info = context.general_context->root_info;
  Node* root = lookup_table.get_node(root_info.node_index);
  if (root) {
    root->stats().RN = std::max(root->stats().RN, 1);
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
                                                   LocalPolicyArray& pi_arr, const Node* node,
                                                   const Edge* edge) {
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
    int updated_edge_arr_index = -1;
    bool cp_has_winning_move = false;
    int num_expanded_edges = 0;

    // read child stats and pi values into arrays to avoid repeated locking
    LocalPolicyArray prior_pi_arr(num_valid_actions);
    util::Gaussian1D prior_logit_beliefs_arr[num_valid_actions];
    util::Gaussian1D cur_logit_beliefs_arr[num_valid_actions];
    LocalActionValueArray child_Q_arr(num_valid_actions, kNumPlayers);
    LocalActionValueArray child_W_arr(num_valid_actions, kNumPlayers);

    for (int i = 0; i < num_valid_actions; i++) {
      const Edge* child_edge = lookup_table.get_edge(node, i);  // TODO: make a safe copy
      const Node* child = lookup_table.get_node(child_edge->child_index);
      prior_pi_arr[i] = child_edge->policy_prior_prob;
      if (child) {
        const auto child_stats = child->stats_safe();  // make a copy

        prior_logit_beliefs_arr[i] = child->stable_data().prior_logit_value_beliefs[seat];
        cur_logit_beliefs_arr[i] = child_stats.logit_value_beliefs[seat];
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
      } else {
        prior_logit_beliefs_arr[i] = child_edge->child_logit_value_beliefs[seat];
        cur_logit_beliefs_arr[i] = child_edge->child_logit_value_beliefs[seat];
        child_Q_arr.row(i) = child_edge->child_AV;
        child_W_arr.row(i) = child_edge->child_AU;
      }
    }

    bool all_edges_expanded = (num_expanded_edges == num_valid_actions);
    if (!all_edges_expanded) {
      all_provably_winning.reset();
      all_provably_losing.reset();
    }

    RELEASE_ASSERT(updated_edge_arr_index >= 0);

    Derived::update_policy(context, pi_arr, node, edge, lookup_table, updated_edge_arr_index,
                           prior_pi_arr, prior_logit_beliefs_arr, cur_logit_beliefs_arr);

    // TODO(1): compute shock terms for action selection

    stats.Q = (child_Q_arr.matrix().transpose() * pi_arr.matrix()).array();
    stats.W = (child_W_arr.matrix().transpose() * pi_arr.matrix()).array();

    stats.Q_min = stats.Q_min.cwiseMin(stats.Q);
    stats.Q_max = stats.Q_max.cwiseMax(stats.Q);
    stats.W_max = stats.W_max.cwiseMax(stats.W);

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
  LookupTable& lookup_table, int updated_edge_arr_index, const LocalPolicyArray& prior_pi_arr,
  const util::Gaussian1D* prior_logit_beliefs, const util::Gaussian1D* cur_logit_beliefs) {
  int i = updated_edge_arr_index;
  util::Gaussian1D QW_i = cur_logit_beliefs[i];
  if (QW_i == util::Gaussian1D::neg_inf()) {
    pi_arr[i] = 0.f;
    normalize_policy(pi_arr);
    return;
  } else if (QW_i == util::Gaussian1D::pos_inf()) {
    pi_arr.fill(0.f);
    pi_arr[i] = 1.f;
    return;
  }

  RELEASE_ASSERT(QW_i.valid());

  float old_pi_i = pi_arr[i];
  if (old_pi_i >= 1.f) {
    // all policy mass is already on this action
    return;
  }

  const auto& stable_data = node->stable_data();
  const int n = stable_data.num_valid_actions - 1;  // excluding updated action

  float P_i = prior_pi_arr[i];
  float Q_i = QW_i.mean();
  float W_i = QW_i.variance();
  float V_i = prior_logit_beliefs[i].mean();
  float U_i = prior_logit_beliefs[i].variance();

  LocalPolicyArray pi(n);
  LocalPolicyArray P(n);
  LocalPolicyArray V(n);
  LocalPolicyArray U(n);
  LocalPolicyArray Q(n);
  LocalPolicyArray W(n);

  int r = 0;
  int w = 0;
  for (; r < n + 1; ++r) {
    if (r == i) {
      continue;
    }
    util::Gaussian1D QW_r = cur_logit_beliefs[r];
    if (QW_r == util::Gaussian1D::pos_inf()) {
      pi_arr.fill(0.f);
      pi_arr[w] = 1.f;
      return;
    }

    pi[w] = pi_arr[r];
    P[w] = prior_pi_arr[r];
    V[w] = prior_logit_beliefs[r].mean();
    U[w] = prior_logit_beliefs[r].variance();
    Q[w] = cur_logit_beliefs[r].mean();
    W[w] = cur_logit_beliefs[r].variance();
    ++w;
  }

  LocalPolicyArray S(n);
  LocalPolicyArray c(n);
  LocalPolicyArray z(n);
  LocalPolicyArray tau(n);

  c = (V_i - V) / (U_i + U).sqrt();
  math::fast_coarse_batch_inverse_normal_cdf_clamped_range(P_i, P.data(), c.data(), n, z.data());
  z -= c;

  S = (Q_i - Q) / (W_i + W).sqrt() + z;
  math::fast_coarse_batch_normal_cdf(S.data(), n, tau.data());

  LocalPolicyArray original_pi_arr = pi_arr;

  pi_arr[i] = (pi * tau).sum() / old_pi_i;
  normalize_policy(pi_arr);

  if (!search::kEnableSearchDebug) return;

  group::element_t inv_sym = Game::SymmetryGroup::inverse(context.leaf_canonical_sym);

  LocalPolicyArray actions(n + 1);
  LocalPolicyArray P2(n + 1);
  LocalPolicyArray V2(n + 1);
  LocalPolicyArray U2(n + 1);
  LocalPolicyArray Q2(n + 1);
  LocalPolicyArray W2(n + 1);
  LocalPolicyArray z2(n + 1);
  LocalPolicyArray tau2(n + 1);

  r = 0;
  for (; r < n + 1; ++r) {
    Edge* child_edge = lookup_table.get_edge(node, r);
    core::action_t action = child_edge->action;
    Game::Symmetries::apply(action, inv_sym, node->action_mode());
    actions(r) = action;
    P2(r) = prior_pi_arr[r];
    V2(r) = prior_logit_beliefs[r].mean();
    U2(r) = prior_logit_beliefs[r].variance();
    Q2(r) = cur_logit_beliefs[r].mean();
    W2(r) = cur_logit_beliefs[r].variance();
    z2(r) = (r == i) ? 0.f : z(r < i ? r : r - 1);
    tau2(r) = (r == i) ? 0.f : tau(r < i ? r : r - 1);
  }

  LOG_INFO("*** DBG update_policy() ***");

  std::stringstream ss;
  auto data = eigen_util::sort_rows(eigen_util::concatenate_columns(actions, P2, V2, U2, Q2, W2, z2,
                                                                    tau2, original_pi_arr, pi_arr));

  std::vector<std::string> columns = {"action", "P", "V",   "U",      "Q",
                                      "W",      "z", "tau", "old_pi", "new_pi"};

  eigen_util::PrintArrayFormatMap fmt_map{
    {"action",
     [&](float x) {
       return Game::IO::action_to_str(x, node->action_mode()) + (x == i ? "*" : "");
     }},
  };

  eigen_util::print_array(ss, data, columns, &fmt_map);
  LOG_INFO("{}", ss.str());
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::populate_logit_value_beliefs(
  const ValueArray& Q, const ValueArray& W, LogitValueArray& logit_value_beliefs) {
  if (kNumPlayers == 2) {
    // In this case, we only need to compute for one player, since the other is just negation.
    logit_value_beliefs[0] = compute_logit_value_belief(Q[0], W[0]);
    logit_value_beliefs[1] = -logit_value_beliefs[0];
  } else {
    for (core::seat_index_t p = 0; p < kNumPlayers; ++p) {
      logit_value_beliefs[p] = compute_logit_value_belief(Q[p], W[p]);
    }
  }
}

template <search::concepts::Traits Traits, typename Derived>
util::Gaussian1D AlgorithmsBase<Traits, Derived>::compute_logit_value_belief(float Q, float W) {
  constexpr float kMin = Game::GameResults::kMinValue;
  constexpr float kMax = Game::GameResults::kMaxValue;
  constexpr float kWidth = kMax - kMin;
  constexpr float kInvWidth = 1.0f / kWidth;

  if (Q <= kMin) {
    return util::Gaussian1D::neg_inf();
  } else if (Q >= kMax) {
    return util::Gaussian1D::pos_inf();
  }

  float mu = Q;
  float sigma_sq = W;

  // Rescale Q and W to reflect [0, 1] range
  mu = (mu - kMin) * kInvWidth;
  sigma_sq *= kInvWidth * kInvWidth;

  float mult = 1.0f / (mu * mu * (1 - mu) * (1 - mu));

  float theta1 = math::fast_coarse_logit(mu);
  float theta2 = (0.5 - mu) * sigma_sq * mult;
  float theta = theta1 - theta2;

  float omega_sq = sigma_sq * mult;
  return util::Gaussian1D(theta, omega_sq);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::normalize_policy(LocalPolicyArray& pi_arr) {
  // renormalize pi_arr
  float pi_sum = pi_arr.sum();
  if (pi_sum > 0.f) {
    pi_arr /= pi_sum;
  } else {
    std::ostringstream ss;
    ss << pi_arr;
    throw util::Exception("Invalid policy: {}", ss.str());
  }
}

}  // namespace beta0
