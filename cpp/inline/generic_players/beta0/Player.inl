#include "generic_players/beta0/Player.hpp"

namespace generic::beta0 {

template <search::concepts::Traits Traits>
typename Player<Traits>::PolicyTensor Player<Traits>::get_action_policy(
  const SearchResults* mcts_results, const ActionMask& valid_actions) const {
  PolicyTensor policy;
  if (this->search_mode_ == core::kRawPolicy) {
    this->raw_init(mcts_results, valid_actions, policy);
  } else {
    policy = mcts_results->P;
  }

  if (this->search_mode_ != core::kRawPolicy) {
    policy = mcts_results->action_symmetry_table.collapse(policy);
    this->apply_temperature(policy);
    policy = mcts_results->action_symmetry_table.symmetrize(policy);

    // TODO: LCB method
  }

  this->normalize(valid_actions, policy);
  return policy;
}

}  // namespace generic::beta0
