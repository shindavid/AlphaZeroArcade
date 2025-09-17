#include "betazero/Algorithms.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
void Algorithms<Traits>::write_to_training_info(const TrainingInfoParams& params,
                                                TrainingInfo& training_info) {
  Base::write_to_training_info(params, training_info);

  const SearchResults* mcts_results = params.mcts_results;
  core::seat_index_t seat = params.seat;

  training_info.Q_prior = Game::GameResults::to_value_array(mcts_results->value_prior)[seat];
  training_info.Q_posterior = mcts_results->win_rates(seat);

  // TODO: we should write action_value_uncertainties_target here as well
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::to_record(const TrainingInfo& training_info,
                                   GameLogFullRecord& full_record) {
  Base::to_record(training_info, full_record);

  full_record.Q_prior = training_info.Q_prior;
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
  compact_record.Q_prior = full_record.Q_prior;
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
void Algorithms<Traits>::to_view(const GameLogCompactRecord&, GameLogView&) {}

}  // namespace beta0
