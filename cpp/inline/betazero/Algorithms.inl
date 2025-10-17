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
void AlgorithmsBase<Traits, Derived>::backprop_helper(Node* node, Edge* edge,
                                                      LookupTable& lookup_table,
                                                      MutexProtectedFunc&& func) {
  mit::unique_lock lock(node->mutex());
  NodeStats old_stats = node->stats();  // make a copy
  func();
  lock.unlock();

  Derived::update_stats(node, edge, lookup_table, old_stats);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::load_evaluations(SearchContext& context) {
  Base::load_evaluations(context);  // assumes that heads[:3] are [policy, value, action-value]

  const LookupTable& lookup_table = context.general_context->lookup_table;
  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());

    auto& stable_data = node->stable_data();
    auto eval = item.eval();

    int n = stable_data.num_valid_actions;

    ValueArray U;
    LocalActionValueArray child_U(n);

    // assumes that heads[3:4] are [value-uncertainty, action-value-uncertainty]
    //
    // TODO: we should be able to verify this assumption at compile-time
    std::copy_n(eval->data(3), U.size(), U.data());
    std::copy_n(eval->data(4), child_U.size(), child_U.data());

    stable_data.U = U;

    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->child_U_estimate = child_U[i];
      edge->policy_posterior_prob = edge->policy_prior_prob;  // initialize posterior to prior
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
  core::seat_index_t seat = root->stable_data().active_seat;

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
    results.action_value_uncertainties(action) = child->stable_data().U[seat];
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

  results.win_rates = stats.Q;
  results.value_prior = stable_data.R;
  results.action_mode = mode;

  results.min_win_rates = stats.Q_min;
  results.max_win_rates = stats.Q_max;

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

  TensorData policy(full_record.policy_target_valid, full_record.policy_target);
  TensorData action_values(full_record.action_values_valid, full_record.action_values);
  TensorData action_value_uncertainties(full_record.action_value_uncertainties_valid,
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
  const TensorData* policy_data = reinterpret_cast<const TensorData*>(policy_data_addr);

  const char* action_values_data_addr = policy_data_addr + policy_data->size();
  const TensorData* action_values_data =
    reinterpret_cast<const TensorData*>(action_values_data_addr);

  const char* action_values_uncertainty_data_addr =
    action_values_data_addr + action_values_data->size();
  const TensorData* action_values_uncertainty_data =
    reinterpret_cast<const TensorData*>(action_values_uncertainty_data_addr);

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
void AlgorithmsBase<Traits, Derived>::update_stats(Node* node, Edge* edge,
                                                   LookupTable& lookup_table,
                                                   const NodeStats& old_stats) {
  ValueArray piW_sum;
  ValueArray piQ_sum;
  ValueArray piQ_sq_sum;
  piW_sum.setZero();
  piQ_sum.setZero();
  piQ_sq_sum.setZero();
  int N = 0;

  // provable bits are maintained for now because we are having beta0 reuse alpha0's selection
  // criterion, which relies on them. Later, we will implement a new selection criterion for beta0,
  // and we can remove the provable bits.
  player_bitset_t all_provably_winning;
  player_bitset_t all_provably_losing;
  all_provably_winning.set();
  all_provably_losing.set();

  auto& stats = node->stats();
  const auto& stable_data = node->stable_data();

  int num_valid_actions = stable_data.num_valid_actions;
  core::seat_index_t seat = stable_data.active_seat;

  if (stable_data.is_chance_node) {
    throw util::Exception("chance nodes not yet supported in beta0");
  } else {
    // read child stats and pi values into arrays to avoid repeated locking
    NodeStats child_stats_arr[num_valid_actions];
    float pi_raw_arr[num_valid_actions];
    int num_expanded_edges = 0;
    int updated_edge_arr_index = -1;

    for (int i = 0; i < num_valid_actions; i++) {
      const Edge* child_edge = lookup_table.get_edge(node, i);
      const Node* child = lookup_table.get_node(child_edge->child_index);
      if (!child) {
        continue;
      }
      child_stats_arr[num_expanded_edges] = child->stats_safe();  // make a copy
      pi_raw_arr[num_expanded_edges] = child_edge->policy_posterior_prob;
      if (child_edge == edge) {
        updated_edge_arr_index = num_expanded_edges;
      }
      num_expanded_edges++;
    }

    Eigen::Map<Eigen::ArrayXf> pi_arr(pi_raw_arr, num_expanded_edges);

    // compute posterior policy
    Derived::update_policy(node, edge, lookup_table, old_stats, child_stats_arr, pi_arr,
                           updated_edge_arr_index);

    // renormalize pi_arr
    float pi_sum = pi_arr.sum();
    if (pi_sum > 0.f) {
      pi_arr *= 1.0f / pi_sum;
    }

    // provably winning/losing calculation
    bool cp_has_winning_move = false;
    for (int i = 0; i < num_expanded_edges; i++) {
      const auto& child_stats = child_stats_arr[i];
      float pi = pi_arr[i];
      piW_sum += child_stats.W * pi;
      piQ_sum += child_stats.Q * pi;
      piQ_sq_sum += child_stats.Q_sq * pi;

      cp_has_winning_move |= child_stats.provably_winning[seat];
      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
    }

    bool all_edges_expanded = (num_expanded_edges == num_valid_actions);
    if (!all_edges_expanded) {
      all_provably_winning.reset();
      all_provably_losing.reset();
    }

    ValueArray V;
    const ValueArray& U = stable_data.U;
    if (stable_data.R_valid) {
      V = Game::GameResults::to_value_array(stable_data.R);
      eigen_util::debug_assert_is_valid_prob_distr(V);
    } else {
      V.setZero();
    }

    ValueArray Q;
    ValueArray Q_sq;

    if (num_expanded_edges || N > 0) {
      ValueArray denom = piW_sum + U * N;

      Q = (V * piW_sum + U * N * piQ_sum) / denom;
      Q_sq = (V * V * piW_sum + U * N * piQ_sq_sum) / denom;
    } else {
      Q.setZero();
      Q_sq.setZero();
    }

    ValueArray W = piW_sum + piQ_sq_sum - Q * Q;
    W = W.cwiseMax(0.f);  // numerical stability

    mit::unique_lock lock(node->mutex());
    if (pi_sum > 0.f) {
      for (int i = 0; i < num_valid_actions; i++) {
        Edge* child_edge = lookup_table.get_edge(node, i);
        child_edge->policy_posterior_prob = pi_arr[i];
      }
    }
    stats.update_q(Q, Q_sq, false);
    stats.W = W;
    stats.update_provable_bits(all_provably_winning, all_provably_losing, cp_has_winning_move,
                               all_edges_expanded, seat);

    if (N) {
      eigen_util::debug_assert_is_valid_prob_distr(stats.Q);
    }
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_policy(
  Node* node, Edge* edge, LookupTable& lookup_table, const NodeStats& old_stats,
  const NodeStats* child_stats_arr, EigenMapArrayXf pi_arr, int updated_edge_arr_index) {
  // Throughout this function, theta and omega_sq represent the mean and variance of the value
  // distribution in an idealized logistic-normal model of the value distribution.

  const auto& stable_data = node->stable_data();
  core::seat_index_t seat = stable_data.active_seat;

  int arr_size = pi_arr.size();
  RELEASE_ASSERT(updated_edge_arr_index >= 0 && updated_edge_arr_index < arr_size);

  double theta_old;
  double omega_sq_old;
  compute_theta_omega_sq(old_stats, seat, theta_old, omega_sq_old);

  double theta_raw_arr[arr_size];
  double omega_sq_raw_arr[arr_size];

  for (int i = 0; i < arr_size; i++) {
    const auto& child_stats = child_stats_arr[i];
    compute_theta_omega_sq(child_stats, seat, theta_raw_arr[i], omega_sq_raw_arr[i]);
  }

  math::finiteness_t finiteness_arr[arr_size];
  bool any_pos_inf = 0;
  for (int i = 0; i < arr_size; i++) {
    finiteness_arr[i] = math::get_finiteness(theta_raw_arr[i]);
    any_pos_inf |= (finiteness_arr[i] == math::kPosInf);
  }

  if (any_pos_inf) {
    // Collapse pi values to only those with +inf theta
    for (int i = 0; i < arr_size; i++) {
      pi_arr[i] = (finiteness_arr[i] == math::kPosInf) ? 1.0f : 0.f;
    }
    return;
  }

  double theta_new = theta_raw_arr[updated_edge_arr_index];
  double omega_sq_new = omega_sq_raw_arr[updated_edge_arr_index];
  math::finiteness_t finiteness_new = finiteness_arr[updated_edge_arr_index];

  EigenMapArrayXd theta_arr(theta_raw_arr, arr_size);
  EigenMapArrayXd omega_sq_arr(omega_sq_raw_arr, arr_size);

  if (finiteness_new == math::kNegInf) {
    // Zero out the just-updated action
    pi_arr[updated_edge_arr_index] = 0;
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

  double alpha_raw_arr[arr_size];
  double beta_raw_arr[arr_size];

  EigenMapArrayXd alpha_arr(alpha_raw_arr, arr_size);
  EigenMapArrayXd beta_arr(beta_raw_arr, arr_size);
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

    double theta_i = theta_raw_arr[i];
    double omega_sq_i = omega_sq_raw_arr[i];

    if (omega_sq_i == 0) {  // certain action
      if (omega_sq_new == 0) {
        RELEASE_ASSERT(theta_new > theta_i);  // expect strict domination
        pi_arr[i] = 0.f;
        continue;
      }
    }

    // Based on all the edge-case checks above, we can now safely compute p and q and expect them
    // to be in (0, 1).
    double p = math::normal_cdf((theta_old - theta_i) / std::sqrt(omega_sq_old + omega_sq_i));
    double q = math::normal_cdf((theta_new - theta_i) / std::sqrt(omega_sq_new + omega_sq_i));

    RELEASE_ASSERT(p > 0.0 && p < 1.0, "invalid p: {}", p);
    RELEASE_ASSERT(q > 0.0 && q < 1.0, "invalid q: {}", q);

    double alpha = q * (1 - p) / (p * (1 - q));
    double beta = pi_arr[i];

    alpha_arr[i] = alpha;
    beta_arr[i] = beta;
  }

  double beta_sum = beta_arr.sum();
  if (beta_sum > 0.0) {
    beta_arr *= (1.0 / beta_sum);
    pi_arr[updated_edge_arr_index] *= std::exp((beta_arr * alpha_arr.log()).sum());
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::compute_theta_omega_sq(const NodeStats& stats,
                                                             core::seat_index_t seat, double& theta,
                                                             double& omega_sq) {
  constexpr double kMin = Game::GameResults::kMinValue;
  constexpr double kMax = Game::GameResults::kMaxValue;

  double mu = stats.Q[seat];

  if (mu == kMin) {
    theta = -std::numeric_limits<double>::infinity();
    omega_sq = 0;
    return;
  } else if (mu == kMax) {
    theta = +std::numeric_limits<double>::infinity();
    omega_sq = 0;
    return;
  }

  mu = (mu - kMin) / (kMax - kMin);  // rescale to [0, 1]
  mu = std::min(std::max(mu, 0.0), 1.0);

  double sigma_sq = stats.W[seat];
  sigma_sq /= (kMax - kMin) * (kMax - kMin);  // rescale to [0, 1]

  // TODO: cache theta/omega_sq values in NodeStats to avoid recomputation

  double theta1 = std::log(mu / (1 - mu));
  double theta2 = -(1 - 2 * mu) * sigma_sq / (2 * mu * mu * (1 - mu) * (1 - mu));
  theta = theta1 + theta2;

  omega_sq = sigma_sq / (mu * mu * (1 - mu) * (1 - mu));
}

}  // namespace beta0
