#include "beta0/Algorithms.hpp"

#include "beta0/Backpropagator.hpp"
#include "core/BasicTypes.hpp"
#include "search/Constants.hpp"
#include "util/EigenUtil.hpp"
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
  stats.Q = q;
  stats.Q_min = stats.Q;
  stats.Q_max = stats.Q;
  stats.W.fill(0.f);
  stats.N = 1;
  populate_logit_value_beliefs(stats.Q, stats.W, stats.lQW);
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
  // search criterion = pi_i * sqrt(W_i) * (N(p) - RC_i)
  const GeneralContext& general_context = *context.general_context;
  const search::SearchParams& search_params = general_context.search_params;
  const LookupTable& lookup_table = general_context.lookup_table;

  Node* node = context.visit_node;
  core::seat_index_t seat = node->stable_data().active_seat;
  int n = node->stable_data().num_valid_actions;

  using Array = LocalPolicyArray;

  Array W(n);
  Array pi(n);
  Array XC(n);
  Array RC(n);
  Array S(n);
  Array score_sq(n);
  int xi = -1;
  int N;

  mit::unique_lock lock(node->mutex());
  N = node->stats().N;
  for (int i = 0; i < n; ++i) {
    Edge* edge = lookup_table.get_edge(node, i);
    Node* child = lookup_table.get_node(edge->child_index);
    if (child) {
      W(i) = child->stats().W[seat];
    } else {
      W(i) = edge->child_AU[seat];
    }
    pi(i) = edge->pi;
    XC(i) = edge->XC;
    RC(i) = edge->RC;
    if (edge->XC > 0 && xi < 0) {
      xi = i;
    }
  }
  lock.unlock();

  S = N - RC;

  int argmax_index;
  if (search_params.tree_size_limit == 1) {
    // net-only, use pi
    pi.maxCoeff(&argmax_index);
  } else {
    if (xi >= 0) {
      argmax_index = xi;
    } else {
      score_sq = W * pi * pi * S * S;
      score_sq.maxCoeff(&argmax_index);
    }

    if (search::kEnableSearchDebug) {
      Array score(n);
      Array actions(n);
      Array argmax(n);
      Array sqrt_W = W.cwiseSqrt();

      if (xi >= 0) {
        score.fill(0.f);
        score(xi) = 1.f;
      } else {
        score = score_sq.cwiseSqrt();
      }

      group::element_t inv_sym = Game::SymmetryGroup::inverse(context.leaf_canonical_sym);
      for (int e = 0; e < n; ++e) {
        auto edge = lookup_table.get_edge(node, e);
        core::action_t action = edge->action;
        Game::Symmetries::apply(action, inv_sym, node->action_mode());
        actions(e) = action;
      }
      argmax.setZero();
      argmax(argmax_index) = 1;

      std::ostringstream ss;
      ss << std::format("{:>{}}", "", context.log_prefix_n());

      static std::vector<std::string> action_columns = {"action", "sqrt(W)", "pi",
                                                        "S",      "score",   "argmax"};
      auto action_data = eigen_util::sort_rows(
        eigen_util::concatenate_columns(actions, sqrt_W, pi, S, score, argmax));

      eigen_util::PrintArrayFormatMap fmt_map{
        {"action", [&](float x) { return Game::IO::action_to_str(x, node->action_mode()); }},
        {"argmax", [](float x) { return std::string(x == 0 ? "" : "*"); }},
      };

      std::stringstream ss2;
      eigen_util::print_array(ss2, action_data, action_columns, &fmt_map);

      std::string line_break =
        std::format("\n{:>{}}", "", util::Logging::kTimestampPrefixLength + context.log_prefix_n());
      for (const std::string& line : util::splitlines(ss2.str())) {
        ss << line << line_break;
      }

      LOG_INFO(ss.str());
    }
  }

  return argmax_index;
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
  results.P.setZero();
  results.pi.setZero();
  results.AQ.setZero();
  results.AW.setZero();

  core::action_t actions[stable_data.num_valid_actions];

  int i = 0;
  for (core::action_t action : stable_data.valid_action_mask.on_indices()) {
    Game::Symmetries::apply(action, inv_sym, mode);
    results.valid_actions.set(action, true);
    actions[i] = action;

    const Edge* edge = lookup_table.get_edge(root, i);
    const Node* child = lookup_table.get_node(edge->child_index);

    results.P(action) = edge->P;
    results.pi(action) = edge->pi;

    if (child) {
      results.AQ.chip(action, 0) = eigen_util::reinterpret_as_tensor(child->stats().Q);
      results.AW.chip(action, 0) = eigen_util::reinterpret_as_tensor(child->stats().W);
    } else {
      results.AQ.chip(action, 0) = eigen_util::reinterpret_as_tensor(edge->child_AV);
      results.AW.chip(action, 0) = eigen_util::reinterpret_as_tensor(edge->child_AU);
    }

    i++;
  }

  results.R = stable_data.R;
  results.Q = stats.Q;
  results.Q_min = stats.Q_min;
  results.Q_max = stats.Q_max;
  results.W = stats.W;

  Derived::load_action_symmetries(general_context, root, &actions[0], results);
  results.action_mode = mode;
  results.trivial = root->trivial();
  results.provably_lost = stats.Q[seat] == Game::GameResults::kMinValue && stats.W[seat] == 0.f;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::write_to_training_info(const TrainingInfoParams& params,
                                                             TrainingInfo& training_info) {
  const SearchResults* mcts_results = params.mcts_results;

  bool use_for_training = params.use_for_training;
  bool previous_used_for_training = params.previous_used_for_training;
  core::seat_index_t seat = params.seat;

  training_info.state = params.state;
  training_info.active_seat = seat;
  training_info.action = params.action;
  training_info.use_for_training = use_for_training;

  if (use_for_training || previous_used_for_training) {
    training_info.policy_target = mcts_results->pi;
    training_info.policy_target_valid =
      Derived::validate_and_symmetrize_policy_target(mcts_results, training_info.policy_target);
  }
  if (use_for_training) {
    training_info.action_values_target = mcts_results->AQ;
    training_info.action_values_target_valid = true;
  }

  training_info.Q = eigen_util::reinterpret_as_tensor(mcts_results->Q);
  training_info.Q_min = eigen_util::reinterpret_as_tensor(mcts_results->Q_min);
  training_info.Q_max = eigen_util::reinterpret_as_tensor(mcts_results->Q_max);
  training_info.W = eigen_util::reinterpret_as_tensor(mcts_results->W);

  if (params.use_for_training) {
    training_info.AW = mcts_results->AW;
    training_info.AW_valid = true;
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::to_record(const TrainingInfo& training_info,
                                                GameLogFullRecord& full_record) {
  Base::to_record(training_info, full_record);

  full_record.Q = training_info.Q;
  full_record.Q_min = training_info.Q_min;
  full_record.Q_max = training_info.Q_max;
  full_record.W = training_info.W;

  if (training_info.AW_valid) {
    full_record.AW = training_info.AW;
  } else {
    full_record.AW.setZero();
  }
  full_record.AW_valid = training_info.AW_valid;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::serialize_record(const GameLogFullRecord& full_record,
                                                       std::vector<char>& buf) {
  GameLogCompactRecord compact_record;
  compact_record.position = full_record.position;
  compact_record.Q = full_record.Q;
  compact_record.Q_min = full_record.Q_min;
  compact_record.Q_max = full_record.Q_max;
  compact_record.W = full_record.W;
  compact_record.active_seat = full_record.active_seat;
  compact_record.action_mode = Game::Rules::get_action_mode(full_record.position);
  compact_record.action = full_record.action;

  PolicyTensorData policy(full_record.policy_target_valid, full_record.policy_target);
  ActionValueTensorData action_values(full_record.action_values_valid, full_record.action_values);
  ActionValueTensorData AW(full_record.AW_valid, full_record.AW);

  search::GameLogCommon::write_section(buf, &compact_record, 1, false);
  policy.write_to(buf);
  action_values.write_to(buf);
  AW.write_to(buf);
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

  const char* AW_data_addr = action_values_data_addr + action_values_data->size();
  const ActionValueTensorData* AW_data =
    reinterpret_cast<const ActionValueTensorData*>(AW_data_addr);

  view.policy_valid = policy_data->load(view.policy);
  view.action_values_valid = action_values_data->load(view.action_values);
  view.AW_valid = AW_data->load(view.AW);

  if (view.policy_valid) {
    Game::Symmetries::apply(view.policy, sym, mode);
  }

  if (view.action_values_valid) {
    Game::Symmetries::apply(view.action_values, sym, mode);
  }

  if (view.AW_valid) {
    Game::Symmetries::apply(view.AW, sym, mode);
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
  view.Q = record->Q;
  view.Q_min = record->Q_min;
  view.Q_max = record->Q_max;
  view.W = record->W;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::populate_logit_value_beliefs(const ValueArray& Q,
                                                                   const ValueArray& W,
                                                                   LogitValueArray& lQW) {
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
  if (W == 0) {
    float theta = math::fast_coarse_logit((Q - kMin) * kInvWidth);
    return util::Gaussian1D(theta, 0.f);
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
