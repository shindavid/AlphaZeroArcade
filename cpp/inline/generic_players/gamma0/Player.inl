#include "generic_players/gamma0/Player.hpp"

#include "core/Constants.hpp"
#include "util/Asserts.hpp"

#include <unistd.h>

namespace generic::gamma0 {

template <search::concepts::Traits Traits>
typename Player<Traits>::ActionResponse Player<Traits>::get_action_response_helper(
  const SearchResults* mcts_results, const ActionRequest& request) {
  PolicyTensor modified_policy = this->get_action_policy(mcts_results, request.valid_actions);

  if (this->verbose_info_) {
    this->verbose_info_->set(modified_policy, *mcts_results);
    VerboseManager::get_instance()->set(this->verbose_info_);
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
  }

  this->normalize(valid_actions, policy);
  return policy;
}

}  // namespace generic::gamma0
