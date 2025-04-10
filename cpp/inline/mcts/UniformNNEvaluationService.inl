#include <mcts/UniformNNEvaluationService.hpp>

namespace mcts {

template <core::concepts::Game Game>
NNEvaluationResponse UniformNNEvaluationService<Game>::evaluate(NNEvaluationRequest& request) {
  for (auto& item : request.stale_items()) {
    this->free_eval(item.eval());
  }
  request.clear_stale_items();

  for (auto& item : request.fresh_items()) {
    const ActionMask& valid_actions = item.node()->stable_data().valid_action_mask;
    auto eval = this->alloc_eval();
    eval->uniform_init(valid_actions);
    item.set_eval(eval);
  }

  return NNEvaluationResponse(0, core::kContinue);
}

}  // namespace mcts
