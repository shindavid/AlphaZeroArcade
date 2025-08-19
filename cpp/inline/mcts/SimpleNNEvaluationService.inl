#include "mcts/SimpleNNEvaluationService.hpp"

namespace mcts {

template <typename Traits>
SimpleNNEvaluationService<Traits>::SimpleNNEvaluationService() {
  eval_pool_.set_recycle_func([](NNEvaluation* eval) { eval->clear(); });
}

template <typename Traits>
core::yield_instruction_t SimpleNNEvaluationService<Traits>::evaluate(
  NNEvaluationRequest& request) {
  mit::unique_lock lock(mutex_);

  for (auto& item : request.stale_items()) {
    eval_pool_.free(item.eval());
  }
  request.clear_stale_items();

  for (auto& item : request.fresh_items()) {
    NNEvaluation* eval = eval_pool_.alloc();
    init_func_(eval, item);
    item.set_eval(eval);
  }

  return core::kContinue;
}

}  // namespace mcts
