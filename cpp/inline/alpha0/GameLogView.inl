#include "alpha0/GameLogView.hpp"

#include "alpha0/GameLogCompactRecord.hpp"
#include "search/GameLogCommon.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
GameLogView<Spec>::GameLogView(const Params& params) {
  using Symmetries = Spec::Symmetries;
  using PolicyShape = TensorEncodings::PolicyEncoding::Shape;
  using ActionValueShape = TensorEncodings::ActionValueEncoding::Shape;
  using PolicyTensorData = search::TensorData<PolicyShape>;
  using ActionValueTensorData = search::TensorData<ActionValueShape>;

  const GameLogCompactRecord* record = params.record;
  const GameLogCompactRecord* next_record = params.next_record;
  const InputFrame* p_cur_frame = params.cur_frame;
  const InputFrame* p_final_frame = params.final_frame;
  const GameResultTensor* outcome = params.outcome;
  group::element_t sym = params.sym;

  active_seat = record->active_seat;

  const char* addr = reinterpret_cast<const char*>(record);

  const char* policy_data_addr = addr + sizeof(GameLogCompactRecord);
  const PolicyTensorData* policy_data = reinterpret_cast<const PolicyTensorData*>(policy_data_addr);

  const char* action_values_data_addr = policy_data_addr + policy_data->size();
  const ActionValueTensorData* action_values_data_ptr =
    reinterpret_cast<const ActionValueTensorData*>(action_values_data_addr);

  policy_valid = policy_data->load(policy);
  action_values_valid = action_values_data_ptr->load(action_values);

  if (policy_valid) {
    Symmetries::apply(policy, sym, *p_cur_frame);
  }

  if (action_values_valid) {
    Symmetries::apply(action_values, sym, *p_cur_frame);
  }

  next_policy_valid = false;
  if (next_record) {
    const char* next_addr = reinterpret_cast<const char*>(next_record);

    const char* next_policy_data_addr = next_addr + sizeof(GameLogCompactRecord);
    const PolicyTensorData* next_policy_data =
      reinterpret_cast<const PolicyTensorData*>(next_policy_data_addr);

    next_policy_valid = next_policy_data->load(next_policy);
    if (next_policy_valid) {
      Symmetries::apply(next_policy, sym, next_record->frame);
    }
  }

  cur_frame = *p_cur_frame;
  final_frame = *p_final_frame;
  game_result = *outcome;
}

}  // namespace alpha0
