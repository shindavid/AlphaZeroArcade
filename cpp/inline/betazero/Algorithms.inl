#include "betazero/Algorithms.hpp"

#include "betazero/Constants.hpp"
#include "core/BasicTypes.hpp"
#include "util/Asserts.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/Gaussian1D.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <cmath>

namespace beta0 {

namespace detail {

// Map (Q, lW) -> lQ, with lW held fixed.
inline float logit_normal_mean(float Q, float lW) {
  // Assumes Q in (0,1).
  const float logit = math::fast_coarse_logit(Q);
  return logit - 0.5f * lW * (1.0f - 2.0f * Q);
}

// Given current Q, fixed lW, and a desired delta_lQ,
// solve for new Q' such that lQ(Q', lW) = lQ(Q, lW) + delta_lQ.
// Uses a few Newton iterations starting from a linearized guess.
inline float newton_update_Q_from_delta_lQ(float Q, float lW, float delta_lQ) {
  // Clamp away from 0/1 for numerical safety.
  constexpr float kEps = 1e-6f;
  constexpr int kMaxIters = 6;
  constexpr float kTol = 1e-3f;

  Q = std::clamp(Q, kEps, 1.0f - kEps);

  // Target lQ after the change.
  const float lQ_current = logit_normal_mean(Q, lW);
  const float lQ_target = lQ_current + delta_lQ;

  // Derivative d(lQ)/dQ at the current Q, with lW fixed:
  //   d/dQ logit = 1/(Q(1-Q))
  //   d/dQ[-(lW/2)(1-2Q)] = lW
  const float dlQ_dQ0 = 1.0f / (Q * (1.0f - Q)) + lW;

  // Linearized initial guess.
  float x = Q + delta_lQ / dlQ_dQ0;
  x = std::clamp(x, kEps, 1.0f - kEps);

  for (int iter = 0; iter < kMaxIters; ++iter) {
    const float lQ = logit_normal_mean(x, lW);

    // f(x) = lQ(x) - lQ_target
    const float fx = lQ - lQ_target;

    // Check convergence in lQ space.
    if (std::fabs(fx) < kTol) {
      break;
    }

    // f'(x) = d(lQ)/dQ at x with lW fixed.
    const float dlQ_dQ = 1.0f / (x * (1.0f - x)) + lW;

    // Guard against pathological derivative (shouldn't really happen for Q in (0,1)).
    if (std::fabs(dlQ_dQ) < 1e-8f) {
      break;
    }

    // Newton step: x_{n+1} = x_n - f(x_n)/f'(x_n).
    float step = fx / dlQ_dQ;
    x -= step;

    x = std::clamp(x, kEps, 1.0f - kEps);

    // Optional: stop if Q itself is barely moving.
    if (std::fabs(step) < kTol * 0.1f) {
      break;
    }
  }

  return x;
}

}  // namespace detail

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

  float minus_shock = 0;
  float plus_shock = 0;
  Derived::update_stats(context_, stats, pi_arr, minus_shock, plus_shock, node, edge);

  lock.lock();
  for (int i = 0; i < num_valid_actions; i++) {
    Edge* child_edge = lookup_table.get_edge(node, i);
    child_edge->policy_posterior_prob = pi_arr[i];

    if (edge == child_edge) {
      child_edge->minus_shock = minus_shock;
      child_edge->plus_shock = plus_shock;
    } else {
      child_edge->minus_shock *= beta0::kShockMult;
      child_edge->plus_shock *= beta0::kShockMult;
    }
  }

  node->stats() = stats;  // copy back
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::init_node_stats_from_terminal(Node* node) {
  Base::init_node_stats_from_terminal(node);

  NodeStats& stats = node->stats();
  populate_logit_value_beliefs(stats.Q, stats.W, stats.logit_value_beliefs);
  stats.minus_shocks.fill(0.f);
  stats.plus_shocks.fill(0.f);
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
    core::seat_index_t seat = stable_data.active_seat;

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

    constexpr float z = beta0::kShockZScore;
    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->policy_prior_prob = P_raw[i];
      edge->adjusted_base_prob = P_adjusted[i];
      edge->child_AU = AU.row(i);
      edge->child_AV = AV.row(i);
      edge->policy_posterior_prob = edge->policy_prior_prob;
      populate_logit_value_beliefs(edge->child_AV, edge->child_AU, edge->child_logit_value_beliefs);
      edge->minus_shock = std::sqrt(edge->child_logit_value_beliefs[seat].variance()) * z;
      edge->plus_shock = edge->minus_shock;
    }

    RELEASE_ASSERT(stats.RN == 0, "RN={}", stats.RN);

    stats.Q = V;

    stats.logit_value_beliefs = stable_data.prior_logit_value_beliefs;
    for (int p = 0; p < kNumPlayers; ++p) {
      stats.minus_shocks[p] = std::sqrt(stats.logit_value_beliefs[p].variance()) * z;
      stats.plus_shocks[p] = stats.minus_shocks[p];
    }

    stats.Q_min = stats.Q_min.cwiseMin(stats.Q);
    stats.Q_max = stats.Q_max.cwiseMax(stats.Q);
    stats.W = U;
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
                                                   LocalPolicyArray& pi_arr, float& minus_shock,
                                                   float& plus_shock, const Node* node,
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
          minus_shock = child_stats.minus_shocks[seat];
          plus_shock = child_stats.plus_shocks[seat];
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

    int i = updated_edge_arr_index;
    RELEASE_ASSERT(i >= 0);

    LocalPolicyArray pi_minus_arr = pi_arr;
    LocalPolicyArray pi_plus_arr = pi_arr;

    Derived::update_policy(context, pi_minus_arr, pi_arr, pi_plus_arr, node, edge, lookup_table,
                           i, prior_pi_arr, prior_logit_beliefs_arr,
                           cur_logit_beliefs_arr, minus_shock, plus_shock);

    Derived::update_QW(&stats.Q, &stats.W, seat, pi_arr, child_Q_arr, child_W_arr);

    if (minus_shock != 0.f) {
      ValueArray prev_Q_i = child_Q_arr.row(i).transpose();
      apply_shock(child_Q_arr, cur_logit_beliefs_arr[i].variance(), -minus_shock, i, seat);
      ValueArray Q_minus;
      Derived::update_QW(&Q_minus, nullptr, seat, pi_minus_arr, child_Q_arr, child_W_arr);
      ValueArray minus_shocks = stats.Q - Q_minus;
      if (minus_shocks[seat] > stats.minus_shocks[seat]) {
        stats.minus_shocks = minus_shocks;
      }
      child_Q_arr.row(i) = prev_Q_i.transpose();
    }

    if (plus_shock != 0.f) {
      apply_shock(child_Q_arr, cur_logit_beliefs_arr[i].variance(), plus_shock, i, seat);
      ValueArray Q_plus;
      Derived::update_QW(&Q_plus, nullptr, seat, pi_plus_arr, child_Q_arr, child_W_arr);
      ValueArray plus_shocks = Q_plus - stats.Q;
      if (plus_shocks[seat] > stats.plus_shocks[seat]) {
        stats.plus_shocks = plus_shocks;
      }
    }

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
  SearchContext& context, LocalPolicyArray& pi_minus_arr, LocalPolicyArray& pi_arr,
  LocalPolicyArray& pi_plus_arr, const Node* node, const Edge* edge, LookupTable& lookup_table,
  int updated_edge_arr_index, const LocalPolicyArray& prior_pi_arr,
  const util::Gaussian1D* prior_logit_beliefs, const util::Gaussian1D* cur_logit_beliefs,
  float minus_shock, float plus_shock) {
  int i = updated_edge_arr_index;
  util::Gaussian1D lQW_i = cur_logit_beliefs[i];
  if (lQW_i == util::Gaussian1D::neg_inf()) {
    pi_minus_arr[i] = 0.f;
    pi_arr[i] = 0.f;
    pi_plus_arr[i] = 0.f;
    normalize_policy(pi_minus_arr);
    normalize_policy(pi_arr);
    normalize_policy(pi_plus_arr);
    return;
  } else if (lQW_i == util::Gaussian1D::pos_inf()) {
    pi_minus_arr.fill(0.f);
    pi_arr.fill(0.f);
    pi_plus_arr.fill(0.f);
    pi_minus_arr[i] = 1.f;
    pi_arr[i] = 1.f;
    pi_plus_arr[i] = 1.f;
    return;
  }

  RELEASE_ASSERT(lQW_i.valid());

  float old_pi_i = pi_arr[i];
  if (old_pi_i >= 1.f) {
    // all policy mass is already on this action
    return;
  }

  const auto& stable_data = node->stable_data();
  const int n = stable_data.num_valid_actions - 1;  // excluding updated action

  float P_i = prior_pi_arr[i];
  float lQ_i = lQW_i.mean();
  float lW_i = lQW_i.variance();
  float lV_i = prior_logit_beliefs[i].mean();
  float lU_i = prior_logit_beliefs[i].variance();

  LocalPolicyArray pi(n);
  LocalPolicyArray P(n);
  LocalPolicyArray lV(n);
  LocalPolicyArray lU(n);
  LocalPolicyArray lQ(n);
  LocalPolicyArray lW(n);

  int r = 0;
  int w = 0;
  for (; r < n + 1; ++r) {
    if (r == i) {
      continue;
    }
    util::Gaussian1D QW_r = cur_logit_beliefs[r];
    if (QW_r == util::Gaussian1D::pos_inf()) {
      pi_minus_arr.fill(0.f);
      pi_arr.fill(0.f);
      pi_plus_arr.fill(0.f);
      pi_minus_arr[w] = 1.f;
      pi_arr[w] = 1.f;
      pi_plus_arr[w] = 1.f;
      return;
    }

    pi[w] = pi_arr[r];
    P[w] = prior_pi_arr[r];
    lV[w] = prior_logit_beliefs[r].mean();
    lU[w] = prior_logit_beliefs[r].variance();
    lQ[w] = cur_logit_beliefs[r].mean();
    lW[w] = cur_logit_beliefs[r].variance();
    ++w;
  }

  LocalPolicyArray original_pi_arr = pi_arr;

  LocalPolicyArray S_denom_inv(n);
  LocalPolicyArray S_minus(n);
  LocalPolicyArray S(n);
  LocalPolicyArray S_plus(n);
  LocalPolicyArray c(n);
  LocalPolicyArray z(n);
  LocalPolicyArray tau_minus(n);
  LocalPolicyArray tau(n);
  LocalPolicyArray tau_plus(n);

  c = (lV_i - lV) / (lU_i + lU).sqrt();
  math::fast_coarse_batch_inverse_normal_cdf_clamped_range(P_i, P.data(), c.data(), n, z.data());
  z -= c;

  S_denom_inv = 1.0f / (lW_i + lW).sqrt();

  S = (lQ_i - lQ) * S_denom_inv + z;
  math::fast_coarse_batch_normal_cdf(S.data(), n, tau.data());
  pi_arr[i] = (pi * tau).sum() / old_pi_i;
  normalize_policy(pi_arr);

  if (minus_shock != 0.f) {
    S_minus = (lQ_i - minus_shock - lQ) * S_denom_inv + z;
    math::fast_coarse_batch_normal_cdf(S_minus.data(), n, tau_minus.data());
    pi_minus_arr[i] = (pi * tau_minus).sum() / old_pi_i;
    normalize_policy(pi_minus_arr);
  } else {
    pi_minus_arr = pi_arr;
  }

  if (plus_shock != 0.f) {
    S_plus = (lQ_i + plus_shock - lQ) * S_denom_inv + z;
    math::fast_coarse_batch_normal_cdf(S_plus.data(), n, tau_plus.data());
    pi_plus_arr[i] = (pi * tau_plus).sum() / old_pi_i;
    normalize_policy(pi_plus_arr);
  } else {
    pi_plus_arr = pi_arr;
  }

  if (!search::kEnableSearchDebug) return;

  group::element_t inv_sym = Game::SymmetryGroup::inverse(context.leaf_canonical_sym);

  LocalPolicyArray actions(n + 1);
  LocalPolicyArray P2(n + 1);
  LocalPolicyArray lV2(n + 1);
  LocalPolicyArray lU2(n + 1);
  LocalPolicyArray lQminus2(n + 1);
  LocalPolicyArray lQ2(n + 1);
  LocalPolicyArray lQplus2(n + 1);
  LocalPolicyArray lW2(n + 1);
  LocalPolicyArray z2(n + 1);
  LocalPolicyArray tau_minus2(n + 1);
  LocalPolicyArray tau2(n + 1);
  LocalPolicyArray tau_plus2(n + 1);

  r = 0;
  for (; r < n + 1; ++r) {
    Edge* child_edge = lookup_table.get_edge(node, r);
    core::action_t action = child_edge->action;
    Game::Symmetries::apply(action, inv_sym, node->action_mode());
    actions(r) = action;
    P2(r) = prior_pi_arr[r];
    lV2(r) = prior_logit_beliefs[r].mean();
    lU2(r) = prior_logit_beliefs[r].variance();
    lQ2(r) = cur_logit_beliefs[r].mean();
    lW2(r) = cur_logit_beliefs[r].variance();
    z2(r) = (r == i) ? 0.f : z(r < i ? r : r - 1);
    tau_minus2(r) = (r == i) ? 0.f : tau_minus(r < i ? r : r - 1);
    tau2(r) = (r == i) ? 0.f : tau(r < i ? r : r - 1);
    tau_plus2(r) = (r == i) ? 0.f : tau_plus(r < i ? r : r - 1);
  }

  lQminus2.setZero();
  lQplus2.setZero();

  lQminus2(i) = lQ_i - minus_shock;
  lQplus2(i) = lQ_i + plus_shock;

  LOG_INFO("*** DBG update_policy() ***");

  std::stringstream ss;
  auto data = eigen_util::sort_rows(eigen_util::concatenate_columns(
    actions, P2, lV2, lU2, lQminus2, lQ2, lQplus2, lW2, z2, tau_minus2, tau2, tau_plus2,
    original_pi_arr, pi_minus_arr, pi_arr, pi_plus_arr));

  std::vector<std::string> columns = {"action", "P",   "lV", "lU",   "lQ-", "lQ",
                                      "lQ+",    "lW",  "z",  "tau-", "tau", "tau+",
                                      "old_pi", "pi-", "pi", "pi+"};

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
void AlgorithmsBase<Traits, Derived>::update_QW(ValueArray* Q, ValueArray* W,
                                                core::seat_index_t seat,
                                                const LocalPolicyArray& pi_arr,
                                                const LocalActionValueArray& child_Q_arr,
                                                const LocalActionValueArray& child_W_arr) {
  // Q/W-Update rules:
  //
  // Q(p) = sum_i pi_i * Q^*_i
  // W(p) = sum_i pi_i (W_i + (Q^*_i - Q(p))^2)
  //
  // where Q^*_i is the *conditional* belief
  //
  // Q^*_i = E[Z_i | i = argmax_j Z_j]
  //
  // We approximate Q^*_i = max(Q_i, Q_floor), where
  //
  // Q_floor is the maximum Q_k over all actions k with W_k = 0 (i.e. no uncertainty).

  float Q_floor = Game::GameResults::kMinValue;
  for (int i = 0; i < child_Q_arr.rows(); ++i) {
    float W_i = child_W_arr(i, seat);
    if (W_i == 0.f) {
      Q_floor = std::max(Q_floor, child_Q_arr(i, seat));
    }
  }

  LocalActionValueArray Q_star_arr = child_Q_arr;
  if (Q_floor > Game::GameResults::kMinValue) {
    // Cap by Q_floor where necessary
    for (int i = 0; i < child_Q_arr.rows(); ++i) {
      if (child_W_arr(i, seat) == 0.f) {
        continue;
      }
      float Q_i = child_Q_arr(i, seat);
      if (Q_i >= Q_floor) {
        continue;
      }
      modify_Q_arr(Q_star_arr, i, seat, Q_floor);
    }
  }

  auto pi_mat = pi_arr.matrix();
  auto Q_star_mat = Q_star_arr.matrix();
  auto Q_p_mat = Q_star_mat.transpose() * pi_mat;
  auto Q_p_arr = Q_p_mat.array();
  *Q = Q_p_arr;

  if (!W) return;

  auto W_in_mat = child_W_arr.matrix();
  auto W_across_mat = (Q_star_arr.rowwise() - Q_p_arr.transpose()).square().matrix();
  auto W_p_mat = (W_in_mat + W_across_mat).transpose() * pi_mat;
  auto W_p_arr = W_p_mat.array();
  *W = W_p_arr;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::apply_shock(LocalActionValueArray& Q_arr, float lW,
                                                  float shock, int action_index,
                                                  core::seat_index_t seat) {
  float old_q = Q_arr(action_index, seat);
  float new_q = detail::newton_update_Q_from_delta_lQ(old_q, lW, shock);
  modify_Q_arr(Q_arr, action_index, seat, new_q);
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

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::modify_Q_arr(LocalActionValueArray& Q_arr, int action_index,
                                                   core::seat_index_t seat, float q_new) {
  int i = action_index;
  float Q_i = Q_arr(i, seat);
  auto row_i = Q_arr.row(i);
  float delta = q_new - Q_i;

  if constexpr (kNumPlayers == 1) {
    // In single-player, we can just adjust the Q value directly.
    row_i(seat) = q_new;
  } else if constexpr (kNumPlayers == 2) {
    // In two-player zero-sum, we can just adjust both players' Q values symmetrically.
    row_i(seat) = q_new;
    row_i(1 - seat) -= delta;
  } else {
    // For multiplayer games, it's a little more ambiguous how we should adjust the other
    // players' Q values.
    //
    // Here, we scale the other players' Q values by a constant chosen such that the sum of Q
    // values remains the same after the adjustment.
    float Q_sum = row_i.sum();
    float mult = (Q_sum - delta) / Q_sum;
    row_i *= mult;
    row_i(seat) = q_new;
  }
}

}  // namespace beta0
