#include <mcts/SimpleNNEvaluationService.hpp>

namespace mcts {

template <core::concepts::Game Game>
SimpleNNEvaluationService<Game>::SimpleNNEvaluationService() {
  eval_pool_.set_recycle_func([](NNEvaluation* eval) {
    eval->clear();
  });
}

template <core::concepts::Game Game>
NNEvaluationResponse SimpleNNEvaluationService<Game>::evaluate(NNEvaluationRequest& request) {
  std::unique_lock lock(mutex_);

  for (auto& item : request.stale_items()) {
    eval_pool_.free(item.eval());
  }
  request.clear_stale_items();

  for (auto& item : request.fresh_items()) {
    NNEvaluation* eval = eval_pool_.alloc();
    init_func_(eval, item);
    item.set_eval(eval);
  }

  return NNEvaluationResponse(0, core::kContinue);
}

}  // namespace mcts
