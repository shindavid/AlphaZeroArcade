#include <mcts/UniformNNEvaluationService.hpp>

namespace mcts {

template <core::concepts::Game Game>
void UniformNNEvaluationService<Game>::evaluate(const NNEvaluationRequest& request) {
  for (typename NNEvaluationRequest::Item& item : request.items()) {
    const ActionMask& valid_actions = item.node()->stable_data().valid_action_mask;
    item.set_eval(NNEvaluation::create_uniform(valid_actions));
  }
}
}  // namespace mcts