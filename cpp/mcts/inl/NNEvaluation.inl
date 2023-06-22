#include <mcts/NNEvaluation.hpp>

#include <util/EigenUtil.hpp>

namespace mcts {

template<core::GameStateConcept GameState>
inline NNEvaluation<GameState>::NNEvaluation(
    const ValueTensor& value, const PolicyTensor& policy, const ActionMask& valid_actions)
{
  GameStateTypes::global_to_local(policy, valid_actions, local_policy_logit_distr_);
  value_prob_distr_ = eigen_util::softmax(eigen_util::reinterpret_as_array(value));
}

}  // namespace mcts
