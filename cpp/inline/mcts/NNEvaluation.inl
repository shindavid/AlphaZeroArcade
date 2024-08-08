#include <mcts/NNEvaluation.hpp>

#include <util/EigenUtil.hpp>
#include <util/Exception.hpp>

namespace mcts {

template <core::concepts::Game Game>
NNEvaluation<Game>::NNEvaluation(const ValueTensor& value, const PolicyTensor& policy,
                                 const ActionMask& valid_actions) {
  int n = valid_actions.count();
  if (n > Game::Constants::kMaxBranchingFactor) {
    throw util::Exception("kMaxBranchingFactor too small (%d < %d)",
                           Game::Constants::kMaxBranchingFactor, n);
  }

  using LocalPolicyArray = Game::Types::LocalPolicyArray;

  LocalPolicyArray local_policy_logit_distr(n);
  int i = 0;
  for (int a : bitset_util::on_indices(valid_actions)) {
    local_policy_logit_distr(i++) = policy(a);
  }

  LocalPolicyArray local_policy_prob_distr = eigen_util::softmax(local_policy_logit_distr);
  compact_local_policy_distr_ = local_policy_prob_distr;
  value_distr_ = eigen_util::softmax(eigen_util::reinterpret_as_array(value));
}

}  // namespace mcts
