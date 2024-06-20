#include <mcts/NNEvaluation.hpp>

#include <util/EigenUtil.hpp>

namespace mcts {

template <core::concepts::Game Game>
inline NNEvaluation<Game>::NNEvaluation(const ValueTensor& value, const PolicyTensor& policy,
                                        const ActionMask& valid_actions) {
  int n_valid_actions = valid_actions.count();
  if (n_valid_actions > Game::Constants::kMaxBranchingFactor) {
    throw util::Exception("kMaxBranchingFactor too small (%d < %d)", Game::Constants::kMaxBranchingFactor,
                          n_valid_actions);
  }
  local_policy_logit_distr_.resize(valid_actions.count());
  const auto& policy_array = eigen_util::reinterpret_as_array(policy);

  int i = 0;
  for (int a : bitset_util::on_indices(valid_actions)) {
    local_policy_logit_distr_(i++) = policy_array(a);
  }

  value_prob_distr_ = eigen_util::softmax(eigen_util::reinterpret_as_array(value));
}

}  // namespace mcts
