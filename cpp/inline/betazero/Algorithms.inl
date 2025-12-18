#include "betazero/Algorithms.hpp"

#include "betazero/Backpropagator.hpp"
#include "core/BasicTypes.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/Gaussian1D.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <cmath>

namespace beta0 {

template <search::concepts::Traits Traits, typename Derived>
template <typename MutexProtectedFunc>
void AlgorithmsBase<Traits, Derived>::backprop(SearchContext& context, Node* node, Edge* edge,
                                               MutexProtectedFunc&& func) {
  if (!edge) {
    mit::unique_lock lock(node->mutex());
    func();
    return;
  }

  using Backpropagator = beta0::Backpropagator<Traits>;
  Backpropagator backpropagator(context, node, edge, func);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::init_node_stats_from_terminal(Node* node) {
  const ValueArray q = Game::GameResults::to_value_array(node->stable_data().R);

  NodeStats& stats = node->stats();
  populate_logit_value_beliefs(stats.Q, stats.W, stats.lQW);
  stats.Q = q;
  stats.Q_min = stats.Q;
  stats.Q_max = stats.Q;
  stats.W.fill(0.f);
  stats.W_max.fill(0.f);
  stats.N = 1;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::init_node_stats_from_nn_eval(Node* node, bool undo_virtual) {
  node->stats().N++;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_node_stats_and_edge(Node* node, Edge* edge,
                                                                 bool undo_virtual) {
  node->stats().N++;
}

template <search::concepts::Traits Traits, typename Derived>
bool AlgorithmsBase<Traits, Derived>::more_search_iterations_needed(
  const GeneralContext& general_context, const Node* root) {
  // root->stats() usage here is not thread-safe but this race-condition is benign
  const search::SearchParams& search_params = general_context.search_params;
  if (!search_params.ponder && root->trivial()) return false;
  return root->stats().N <= search_params.tree_size_limit;
}

template <search::concepts::Traits Traits, typename Derived>
int AlgorithmsBase<Traits, Derived>::get_best_child_index(const SearchContext& context) {
  // TODO: search criterion = pi_i * sqrt(W_i) * (N(p) - RC_i)
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

    ValueArray V = Game::GameResults::to_value_array(R);

    stable_data.R = R;
    stable_data.R_valid = true;
    stable_data.U = U;
    populate_logit_value_beliefs(V, U, stable_data.lUV);

    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->P = P_raw[i];
      edge->child_AU = AU.row(i);
      edge->child_AV = AV.row(i);
      edge->pi = edge->P;

      // TODO: move this outside the loop, and do it as a batch calc off AV and AU, to
      // vectorize the division
      populate_logit_value_beliefs(edge->child_AV, edge->child_AU, edge->child_lAUV);
    }

    stats.Q = V;
    stats.lQW = stable_data.lUV;
    stats.Q_min = stats.Q_min.cwiseMin(stats.Q);
    stats.Q_max = stats.Q_max.cwiseMax(stats.Q);
    stats.W = U;
    stats.W_max = stats.W_max.cwiseMax(stats.W);
  }

  const RootInfo& root_info = context.general_context->root_info;
  Node* root = lookup_table.get_node(root_info.node_index);
  if (root) {
    root->stats().N = std::max(root->stats().N, 1);
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

  core::seat_index_t seat = stable_data.active_seat;
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
    results.policy_prior(action) = edge->P;
    results.policy_posterior(action) = edge->pi;

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
  results.provably_lost = stats.Q[seat] == Game::GameResults::kMinValue && stats.W[seat] == 0.f;
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
void AlgorithmsBase<Traits, Derived>::write_results(const GeneralContext& general_context,
                                                    const Node* root, group::element_t inv_sym,
                                                    SearchResults& results) {
  throw util::CleanException("write_results not yet support in beta0");
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::populate_logit_value_beliefs(
  const ValueArray& Q, const ValueArray& W, LogitValueArray& lQW) {
  if (kNumPlayers == 2) {
    // In this case, we only need to compute for one player, since the other is just negation.
    lQW[0] = compute_logit_value_belief(Q[0], W[0]);
    lQW[1] = -lQW[0];
  } else {
    for (core::seat_index_t p = 0; p < kNumPlayers; ++p) {
      lQW[p] = compute_logit_value_belief(Q[p], W[p]);
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

}  // namespace beta0
