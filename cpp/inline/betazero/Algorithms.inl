#include "betazero/Algorithms.hpp"

#include "util/Asserts.hpp"
#include "util/CppUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/Math.hpp"

#include <limits>

namespace beta0 {

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
  func();
  if (!edge) return;
  NodeStats stats = node->stats();  // copy

  int num_valid_actions = node->stable_data().num_valid_actions;
  LocalPolicyArray pi_arr(num_valid_actions);
  for (int i = 0; i < num_valid_actions; i++) {
    const Edge* child_edge = lookup_table_.get_edge(node, i);
    pi_arr(i) = child_edge->policy_posterior_prob;
  }
  lock.unlock();

  update_stats(stats, pi_arr, node, edge, lookup_table_);

  lock.lock();

  for (int i = 0; i < num_valid_actions; i++) {
    Edge* child_edge = lookup_table_.get_edge(node, i);
    child_edge->policy_posterior_prob = pi_arr[i];
  }

  // Carefully copy back fields of stats back to node->stats()
  // We don't copy counts, which may have been updated by other threads.
  int RN = node->stats().RN;
  int VN = node->stats().VN;
  node->stats() = stats;
  node->stats().RN = RN;
  node->stats().VN = VN;
  lock.unlock();
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
    LocalActionValueArray AU(n, Game::Constants::kNumPlayers);

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

  check_values(results.min_win_rates, __LINE__);
  check_values(results.max_win_rates, __LINE__);
  check_values(results.action_value_uncertainties, __LINE__);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::write_to_training_info(const TrainingInfoParams& params,
                                                             TrainingInfo& training_info) {
  Base::write_to_training_info(params, training_info);

  const SearchResults* mcts_results = params.mcts_results;

  for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
    training_info.Q_posterior(p) = mcts_results->win_rates[p];
    training_info.Q_min(p) = mcts_results->min_win_rates[p];
    training_info.Q_max(p) = mcts_results->max_win_rates[p];
  }

  check_values(training_info.Q_min, __LINE__);
  check_values(training_info.Q_max, __LINE__);

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

  check_values(full_record.Q_min, __LINE__);
  check_values(full_record.Q_max, __LINE__);

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
  compact_record.active_seat = full_record.active_seat;
  compact_record.action_mode = Game::Rules::get_action_mode(full_record.position);
  compact_record.action = full_record.action;

  check_values(compact_record.Q_min, __LINE__);
  check_values(compact_record.Q_max, __LINE__);
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
  Base::to_view(params, view);

  const GameLogCompactRecord* record = params.record;
  group::element_t sym = params.sym;
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

  view.action_value_uncertainties_valid =
    action_values_uncertainty_data->load(view.action_value_uncertainties);

  if (view.action_value_uncertainties_valid) {
    Game::Symmetries::apply(view.action_value_uncertainties, sym, mode);
  }

  view.Q_posterior = record->Q_posterior;
  view.Q_min = record->Q_min;
  view.Q_max = record->Q_max;

  check_values(view.Q_min, __LINE__);
  check_values(view.Q_max, __LINE__);
  check_values(view.action_value_uncertainties, __LINE__);
}


template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_stats(NodeStats& stats, LocalPolicyArray& pi_arr,
                                                   const Node* node, const Edge* edge,
                                                   LookupTable& lookup_table) {
  Base::update_stats(stats, node, lookup_table);  // writes stats.Q and stats.Q_sq

  ValueArray old_child_Qbeta = edge->child_Qbeta_snapshot;
  ValueArray old_child_W = edge->child_W_snapshot;

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
                           old_child_Qbeta[seat], old_child_W[seat], child_Qbeta_arr.col(seat),
                           child_W_arr.col(seat));

    // renormalize pi_arr
    float pi_sum = pi_arr.sum();
    if (pi_sum > 0.f) {
      pi_arr *= 1.0 / pi_sum;
    }

    RELEASE_ASSERT(!pi_arr.hasNaN());

    auto M = child_Qbeta_arr.matrix().transpose();
    ValueArray piQbeta_sum = (M * pi_arr.matrix()).array();
    ValueArray piQbeta_sq_sum = (M * M * pi_arr.matrix()).array();
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
    ValueArray W = piW_sum + piQbeta_sq_sum - Qbeta * Qbeta;
    W = W.cwiseMax(0.f);  // numerical stability

    stats.Qbeta = Qbeta;
    stats.Qbeta_min = stats.Qbeta_min.min(Qbeta);
    stats.Qbeta_max = stats.Qbeta_max.max(Qbeta);
    stats.W = W;
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

  int arr_size = pi_arr.size();
  RELEASE_ASSERT(updated_edge_arr_index >= 0 && updated_edge_arr_index < arr_size);

  float theta_old;
  float omega_sq_old;
  compute_theta_omega_sq(old_child_Qbeta, old_child_W, theta_old, omega_sq_old);

  LocalPolicyArray theta_arr(arr_size);
  LocalPolicyArray omega_sq_arr(arr_size);

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
    return;
  }

  float theta_new = theta_arr[updated_edge_arr_index];
  float omega_sq_new = omega_sq_arr[updated_edge_arr_index];
  math::finiteness_t finiteness_new = finiteness_arr[updated_edge_arr_index];

  if (finiteness_new == math::kNegInf) {
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
          return;
        } else if (theta_arr[i] == theta_new) {
          // tie
          pi_arr[updated_edge_arr_index] = pi_arr[i];
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

  LocalPolicyArray alpha_arr(arr_size);
  LocalPolicyArray beta_arr(arr_size);
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

    float theta_i = theta_arr[i];
    float omega_sq_i = omega_sq_arr[i];

    if (omega_sq_i == 0) {  // certain action
      if (omega_sq_new == 0) {
        RELEASE_ASSERT(theta_new > theta_i);  // expect strict domination
        pi_arr[i] = 0.f;
        continue;
      }
    }

    // Based on all the edge-case checks above, we can now safely compute p and q and expect them
    // to be in (0, 1).
    float p = math::normal_cdf((theta_old - theta_i) / std::sqrt(omega_sq_old + omega_sq_i));
    float q = math::normal_cdf((theta_new - theta_i) / std::sqrt(omega_sq_new + omega_sq_i));

    RELEASE_ASSERT(p > 0.0 && p < 1.0, "invalid p: {}", p);
    RELEASE_ASSERT(q > 0.0 && q < 1.0, "invalid q: {}", q);

    float alpha = q * (1 - p) / (p * (1 - q));
    float beta = pi_arr[i];

    alpha_arr[i] = alpha;
    beta_arr[i] = beta;
  }

  float beta_sum = beta_arr.sum();
  if (beta_sum > 0.0) {
    beta_arr *= (1.0 / beta_sum);
    pi_arr[updated_edge_arr_index] *= std::exp((beta_arr * alpha_arr.log()).sum());
    RELEASE_ASSERT(!std::isnan(pi_arr[updated_edge_arr_index]));
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::compute_theta_omega_sq(float Qbeta, float W, float& theta,
                                                             float& omega_sq) {
  constexpr float kMin = Game::GameResults::kMinValue;
  constexpr float kMax = Game::GameResults::kMaxValue;

  if (Qbeta <= kMin) {
    theta = -std::numeric_limits<float>::infinity();
    omega_sq = 0;
    return;
  } else if (Qbeta >= kMax) {
    theta = +std::numeric_limits<float>::infinity();
    omega_sq = 0;
    return;
  }

  float mu = Qbeta;

  // Rescale Qbeta to [0, 1]
  mu = (mu - kMin) / (kMax - kMin);
  mu = std::max(0.0f, std::min(1.0f, mu));

  // Rescale W to variance in [0, 1]
  float sigma_sq = W;
  sigma_sq /= (kMax - kMin) * (kMax - kMin);
  sigma_sq = std::max(0.0f, std::min(1.0f, sigma_sq));

  // TODO: cache theta/omega_sq values in NodeStats to avoid recomputation

  float theta1 = std::log(mu / (1 - mu));
  float theta2 = -(1 - 2 * mu) * sigma_sq / (2 * mu * mu * (1 - mu) * (1 - mu));
  theta = theta1 + theta2;

  omega_sq = sigma_sq / (mu * mu * (1 - mu) * (1 - mu));
}

}  // namespace beta0
