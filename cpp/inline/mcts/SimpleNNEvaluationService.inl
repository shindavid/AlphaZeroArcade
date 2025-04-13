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

  core::cache_shard_index_t shard_index = 0;
  for (; shard_index < mcts::kNumCacheShards; ++shard_index) {
    auto& sub_request = request.sub_request(shard_index);

    int num_fresh_items = sub_request.num_fresh_items();
    if (num_fresh_items == 0) continue;

    for (auto& item : sub_request.stale_items()) {
      eval_pool_.free(item.eval());
    }
    sub_request.clear_stale_items();

    for (auto& item : sub_request.fresh_items()) {
      NNEvaluation* eval = eval_pool_.alloc();
      init_func_(eval, item);
      item.set_eval(eval);
    }
  }

  return NNEvaluationResponse(0, core::kContinue);
}

}  // namespace mcts
