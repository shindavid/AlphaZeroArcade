#include "search/SimpleNNEvaluationService.hpp"

namespace search {

template <::alpha0::concepts::Spec Spec>
SimpleNNEvaluationService<Spec>::SimpleNNEvaluationService() {
  eval_pool_.set_recycle_func([](NNEvaluation* eval) { eval->clear(); });
}

template <::alpha0::concepts::Spec Spec>
core::yield_instruction_t SimpleNNEvaluationService<Spec>::evaluate(
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

}  // namespace search
