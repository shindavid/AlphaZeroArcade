#include "betazero/Algorithms.hpp"

#include "util/CppUtil.hpp"

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
void AlgorithmsBase<Traits, Derived>::load_evaluations(SearchContext& context) {
  Base::load_evaluations(context);  // assumes that heads[:3] are [policy, value, action-value]

  const LookupTable& lookup_table = context.general_context->lookup_table;
  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());

    auto& stable_data = node->stable_data();
    auto eval = item.eval();

    int n = stable_data.num_valid_actions;

    using ValueUncertaintyTensor = Traits::EvalSpec::NetworkHeads::ValueUncertaintyHead::Tensor;
    ValueUncertaintyTensor U;
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
    }
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::to_results(const GeneralContext& general_context,
                                                 SearchResults& results) {
  Base::to_results(general_context, results);
  const LookupTable& lookup_table = general_context.lookup_table;
  const Node* root = lookup_table.get_node(general_context.root_info.node_index);
  auto& action_value_uncertainties = results.action_value_uncertainties;
  core::seat_index_t seat = root->stable_data().active_seat;

  action_value_uncertainties.setZero();
  for (int i = 0; i < root->stable_data().num_valid_actions; i++) {
    const Edge* edge = lookup_table.get_edge(root, i);
    const Node* child = lookup_table.get_node(edge->child_index);
    if (!child) continue;

    core::action_t action = edge->action;
    const auto& stable_data = child->stable_data();
    action_value_uncertainties(action) = stable_data.U(seat);
  }

  core::action_mode_t mode = root->action_mode();
  group::element_t sym = general_context.root_info.canonical_sym;
  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);
  Game::Symmetries::apply(action_value_uncertainties, inv_sym, mode);

  const auto& stats = root->stats();  // thread-safe since single-threaded here
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
void AlgorithmsBase<Traits, Derived>::init_q(NodeStats& stats, const ValueArray& value, bool pure) {
  stats.init_q(value, pure);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_q(NodeStats& stats, const ValueArray& value) {
  stats.update_q(value);
}

}  // namespace beta0
