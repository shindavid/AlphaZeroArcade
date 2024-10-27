#include <mcts/UniformNNEvaluationService.hpp>

namespace mcts {

template <core::concepts::Game Game>
void UniformNNEvaluationService<Game>::evaluate(const NNEvaluationRequest& request) {
  ValueTensor value;
  PolicyTensor policy;
  ActionValueTensor action_values;
  group::element_t sym = group::kIdentity;

  for (typename NNEvaluationRequest::Item& item : request.items()) {
    ActionMask valid_actions = item.node()->stable_data().valid_action_mask;
    core::seat_index_t cp = item.node()->stable_data().current_player;

    policy.setZero();
    value.setZero();
    action_values.setZero();

    item.set_eval(
        std::make_shared<NNEvaluation>(value, policy, action_values, valid_actions, sym, cp));
  }
}
}  // namespace mcts