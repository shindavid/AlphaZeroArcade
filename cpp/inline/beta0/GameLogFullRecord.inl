#include "beta0/GameLogFullRecord.hpp"

#include "beta0/TrainingInfo.hpp"
#include "search/GameLogCommon.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
GameLogFullRecord<Spec>::GameLogFullRecord(const TrainingInfo<Spec>& info) {
  frame = info.frame;
  Q_root = info.Q_root;

  if (info.policy_target_valid) {
    policy_target = info.policy_target;
  } else {
    policy_target.setZero();
  }

  if (info.action_values_target_valid) {
    action_values = info.action_values_target;
    action_values_uncertainty = info.action_values_uncertainty_target;
  } else {
    action_values.setZero();
    action_values_uncertainty.setZero();
  }

  move = info.move;
  active_seat = info.active_seat;
  use_for_training = info.use_for_training;
  policy_target_valid = info.policy_target_valid;
  action_values_valid = info.action_values_target_valid;
  future_mcts_value_target_valid = false;
  backup_sample = info.backup_sample;
}

template <beta0::concepts::Spec Spec>
void GameLogFullRecord<Spec>::set_future_mcts_value_target(
  const ValueArray& future_mcts_value_target_val) {
  future_mcts_value_target = future_mcts_value_target_val;
  future_mcts_value_target_valid = true;
}

template <beta0::concepts::Spec Spec>
void GameLogFullRecord<Spec>::serialize(std::vector<char>& buf) const {
  GameLogCompactRecord compact_record;
  compact_record.frame = frame;
  compact_record.active_seat = active_seat;
  compact_record.move = move;
  compact_record.future_mcts_value_target = future_mcts_value_target;
  compact_record.future_mcts_value_target_valid = future_mcts_value_target_valid;
  compact_record.backup_sample = backup_sample;

  PolicyTensorData policy(policy_target_valid, policy_target);
  ActionValueTensorData action_values_data(action_values_valid, action_values);
  ActionValueTensorData action_values_uncertainty_data(action_values_valid,
                                                       action_values_uncertainty);

  search::GameLogCommon::write_section(buf, &compact_record, 1, false);
  policy.write_to(buf);
  action_values_data.write_to(buf);
  action_values_uncertainty_data.write_to(buf);
}

}  // namespace beta0
