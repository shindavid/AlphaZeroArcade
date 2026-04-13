#include "alpha0/GameLogFullRecord.hpp"

#include "alpha0/TrainingInfo.hpp"
#include "search/GameLogBase.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
GameLogFullRecord<EvalSpec>::GameLogFullRecord(const TrainingInfo<EvalSpec>& info) {
  frame = info.frame;

  if (info.policy_target_valid) {
    policy_target = info.policy_target;
  } else {
    policy_target.setZero();
  }

  if (info.action_values_target_valid) {
    action_values = info.action_values_target;
  } else {
    action_values.setZero();
  }

  move = info.move;
  active_seat = info.active_seat;
  use_for_training = info.use_for_training;
  policy_target_valid = info.policy_target_valid;
  action_values_valid = info.action_values_target_valid;
}

template <core::concepts::EvalSpec EvalSpec>
void GameLogFullRecord<EvalSpec>::serialize(std::vector<char>& buf) const {
  GameLogCompactRecord compact_record;
  compact_record.frame = frame;
  compact_record.active_seat = active_seat;
  compact_record.move = move;

  PolicyTensorData policy(policy_target_valid, policy_target);
  ActionValueTensorData action_values_data(action_values_valid, action_values);

  search::GameLogCommon::write_section(buf, &compact_record, 1, false);
  policy.write_to(buf);
  action_values_data.write_to(buf);
}

}  // namespace alpha0
