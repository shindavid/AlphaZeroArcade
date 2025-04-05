#include <mcts/UniformNNEvaluationService.hpp>

namespace mcts {

template <core::concepts::Game Game>
NNEvaluationResponse UniformNNEvaluationService<Game>::evaluate(NNEvaluationRequest& request) {
  for (typename NNEvaluationRequest::Item& item : request.items()) {
    const ActionMask& valid_actions = item.node()->stable_data().valid_action_mask;
    item.set_eval(NNEvaluation::create_uniform(valid_actions));
  }

  return NNEvaluationResponse(sequence_id_++, core::kContinue);
}

}  // namespace mcts
