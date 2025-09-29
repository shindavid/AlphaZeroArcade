#include "betazero/Algorithms.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
void Algorithms<Traits>::load_evaluations(SearchContext& context) {
  Base::load_evaluations(context);  // assumes that heads[:3] are [policy, value, action-value]

  const LookupTable& lookup_table = context.general_context->lookup_table;
  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());

    auto& stable_data = node->stable_data();
    auto eval = item.eval();

    int n = stable_data.num_valid_actions;

    using ValueUncertaintyTensor = EvalSpec::NetworkHeads::ValueUncertaintyHead::Tensor;
    ValueUncertaintyTensor U;
    LocalActionValueArray child_U(n);

    // assumes that heads[3:4] are [value-uncertainty, action-value-uncertainty]
    //
    // TODO: we should be able to verify this assumption at compile-time
    std::copy_n(eval->data(3), U.size(), U.data());
    std::copy_n(eval->data(4), child_U.size(), child_U.data());

    stable_data.U = U(0);

    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->child_U_estimate = child_U[i];
    }
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::write_to_training_info(const TrainingInfoParams& params,
                                                TrainingInfo& training_info) {
  Base::write_to_training_info(params, training_info);

  const SearchResults* mcts_results = params.mcts_results;

  for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
    training_info.Q_posterior(p) = mcts_results->win_rates[p];
  }

  if (params.use_for_training) {
    training_info.action_value_uncertainties_target = mcts_results->action_value_uncertainties;
    training_info.action_value_uncertainties_target_valid = true;
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::to_record(const TrainingInfo& training_info,
                                   GameLogFullRecord& full_record) {
  Base::to_record(training_info, full_record);

  full_record.Q_posterior = training_info.Q_posterior;

  if (training_info.action_value_uncertainties_target_valid) {
    full_record.action_value_uncertainties = training_info.action_value_uncertainties_target;
  } else {
    full_record.action_value_uncertainties.setZero();
  }
  full_record.action_value_uncertainties_valid =
    training_info.action_value_uncertainties_target_valid;
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::serialize_record(const GameLogFullRecord& full_record,
                                          std::vector<char>& buf) {
  GameLogCompactRecord compact_record;
  compact_record.position = full_record.position;
  compact_record.Q_posterior = full_record.Q_posterior;
  compact_record.active_seat = full_record.active_seat;
  compact_record.action_mode = Game::Rules::get_action_mode(full_record.position);
  compact_record.action = full_record.action;

  TensorData policy(full_record.policy_target_valid, full_record.policy_target);
  TensorData action_values(full_record.action_values_valid, full_record.action_values);
  TensorData action_value_uncertainties(full_record.action_value_uncertainties_valid,
                                        full_record.action_value_uncertainties);

  search::GameLogCommon::write_section(buf, &compact_record, 1, false);
  policy.write_to(buf);
  action_values.write_to(buf);
  action_value_uncertainties.write_to(buf);
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::to_view(const GameLogViewParams& params, GameLogView& view) {
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
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::to_results(const GeneralContext& general_context, SearchResults& results) {
  Base::to_results(general_context, results);
  const LookupTable& lookup_table = general_context.lookup_table;
  const Node* root = lookup_table.get_node(general_context.root_info.node_index);
  auto& action_value_uncertainties = results.action_value_uncertainties;

  action_value_uncertainties.setZero();
  for (int i = 0; i < root->stable_data().num_valid_actions; i++) {
    const Edge* edge = lookup_table.get_edge(root, i);
    const Node* child = lookup_table.get_node(edge->child_index);
    if (!child) continue;

    core::action_t action = edge->action;
    const auto& stable_data = child->stable_data();
    action_value_uncertainties(action) = stable_data.U;
  }

  core::action_mode_t mode = root->action_mode();
  group::element_t sym = general_context.root_info.canonical_sym;
  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);
  Game::Symmetries::apply(action_value_uncertainties, inv_sym, mode);
}

}  // namespace beta0
