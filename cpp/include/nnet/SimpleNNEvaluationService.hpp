#pragma once

#include "core/BasicTypes.hpp"
#include "nnet/NNEvaluation.hpp"
#include "nnet/NNEvaluationRequest.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"
#include "util/RecyclingAllocPool.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <functional>

namespace nnet {

// SimpleNNEvaluationService is a simple class that implements the NNEvaluationServiceBase
// interface. It is simple in the sense that its evaluate() method never yields. It is only
// suitable for unit-test mocking purposes, and for the UniformNNEvaluationService.
template <typename Traits>
class SimpleNNEvaluationService : public nnet::NNEvaluationServiceBase<Traits> {
 public:
  using Game = Traits::Game;
  using NNEvaluation = nnet::NNEvaluation<Game>;
  using NNEvaluationRequest = nnet::NNEvaluationRequest<Traits>;
  using Item = NNEvaluationRequest::Item;
  using Node = Traits::Node;
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

}  // namespace nnet

#include "inline/nnet/SimpleNNEvaluationService.inl"
