#include "generic_players/beta0/Player.hpp"

#include "core/Constants.hpp"
#include "util/Asserts.hpp"

#include <unistd.h>

namespace generic::beta0 {

template <search::concepts::Traits Traits>
typename Player<Traits>::ActionResponse Player<Traits>::get_action_response_helper(
  const SearchResults* mcts_results, const ActionRequest& request) {
  PolicyTensor modified_policy = this->get_action_policy(mcts_results, request.valid_actions);

  if (this->verbose_info_) {
    this->verbose_info_->action_policy = modified_policy;
    this->verbose_info_->mcts_results = *mcts_results;
    this->verbose_info_->initialized = true;
  }
  core::action_t action = eigen_util::sample(modified_policy);
  RELEASE_ASSERT(request.valid_actions[action]);
  return action;
}

template <search::concepts::Traits Traits>
auto Player<Traits>::get_action_policy(const SearchResults* mcts_results,
                                       const ActionMask& valid_actions) const {
  PolicyTensor policy;
  if (this->search_mode_ == core::kRawPolicy) {
    this->raw_init(mcts_results, valid_actions, policy);
  } else {
    policy = mcts_results->policy_posterior;
    policy = mcts_results->action_symmetry_table.collapse(policy);
    this->apply_temperature(policy);
    policy = mcts_results->action_symmetry_table.symmetrize(policy);
    if (this->params_.LCB_z_score) {
      this->apply_LCB(mcts_results, valid_actions, policy);
    }
  }

  this->normalize(valid_actions, policy);
  return policy;
}

template <search::concepts::Traits Traits>
inline void Player<Traits>::verbose_dump() const {
  if (!verbose_info_->initialized) return;

  const auto& action_policy = verbose_info_->action_policy;
  const auto& mcts_results = verbose_info_->mcts_results;
  int num_rows_to_display = params_.verbose_num_rows_to_display;
}

}  // namespace generic::beta0
