#include "alpha0/TrainingInfo.hpp"

#include "alpha0/SearchResults.hpp"
#include "util/EigenUtil.hpp"

namespace alpha0 {

template <alpha0::concepts::EvalSpec EvalSpec>
TrainingInfo<EvalSpec>::TrainingInfo(bool use_for_training_, const ActionResponse& response,
                                     const SearchResults<EvalSpec>* mcts_results,
                                     core::seat_index_t seat,
                                     bool prev_entry_used_for_training) {
  clear();
  frame = mcts_results->frame;
  active_seat = seat;
  move = response.get_move();
  use_for_training = use_for_training_;

  if (use_for_training_ || prev_entry_used_for_training) {
    policy_target = mcts_results->policy_target;
    policy_target_valid = validate_and_symmetrize_policy_target(mcts_results, policy_target);
  }

  if (use_for_training_) {
    action_values_target = apply_mask(mcts_results->AV, mcts_results->pre_expanded_moves);
    action_values_target_valid = true;
  }
}

template <alpha0::concepts::EvalSpec EvalSpec>
bool TrainingInfo<EvalSpec>::validate_and_symmetrize_policy_target(
  const SearchResults<EvalSpec>* mcts_results, PolicyTensor& target) {
  float sum = eigen_util::sum(target);
  if (mcts_results->provably_lost || sum == 0 || mcts_results->trivial) {
    return false;
  } else {
    target = mcts_results->action_symmetry_table.symmetrize(mcts_results->frame, target);
    target = target / eigen_util::sum(target);
    eigen_util::debug_assert_is_valid_prob_distr(target);
    return true;
  }
}

template <alpha0::concepts::EvalSpec EvalSpec>
typename TrainingInfo<EvalSpec>::ActionValueTensor TrainingInfo<EvalSpec>::apply_mask(
  const ActionValueTensor& values, const PolicyTensor& mask, float invalid_value) {
  using Indices = Eigen::array<Eigen::Index, 2>;
  Indices reshape_dims = {mask.dimensions()[0], 1};
  Indices bcast = {1, values.dimensions()[1]};
  auto reshaped_mask = mask.reshape(reshape_dims).broadcast(bcast);
  auto selector = reshaped_mask > reshaped_mask.constant(0.5f);
  ActionValueTensor invalid_tensor = reshaped_mask.constant(invalid_value);
  return selector.select(values, invalid_tensor);
}

}  // namespace alpha0
