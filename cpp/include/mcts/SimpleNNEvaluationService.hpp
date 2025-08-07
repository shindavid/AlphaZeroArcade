#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/Game.hpp"
#include "mcts/NNEvaluation.hpp"
#include "mcts/NNEvaluationRequest.hpp"
#include "mcts/NNEvaluationServiceBase.hpp"
#include "mcts/Node.hpp"
#include "util/RecyclingAllocPool.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <functional>

namespace mcts {

// SimpleNNEvaluationService is a simple class that implements the NNEvaluationServiceBase
// interface. It is simple in the sense that its evaluate() method never yields. It is only
// suitable for unit-test mocking purposes, and for the UniformNNEvaluationService.
template <core::concepts::Game Game>
class SimpleNNEvaluationService : public mcts::NNEvaluationServiceBase<Game> {
 public:
  using NNEvaluation = mcts::NNEvaluation<Game>;
  using NNEvaluationRequest = mcts::NNEvaluationRequest<Game>;
  using Item = NNEvaluationRequest::Item;
  using Node = mcts::Node<Game>;
  using EvalPool = util::RecyclingAllocPool<NNEvaluation>;
  using init_func_t = std::function<void(NNEvaluation*, const Item&)>;

  SimpleNNEvaluationService();

  // Set the function that will be called to initialize the NNEvaluation object.
  void set_init_func(init_func_t f) { init_func_ = std::move(f); }

  core::yield_instruction_t evaluate(NNEvaluationRequest& request) override;

 private:
  mit::mutex mutex_;
  EvalPool eval_pool_;
  init_func_t init_func_ = [](NNEvaluation*, const Item&) {};
};

}  // namespace mcts

#include "inline/mcts/SimpleNNEvaluationService.inl"
