#pragma once

#include "alpha0/GraphTraits.hpp"
#include "alpha0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"
#include "search/NNEvaluation.hpp"
#include "search/NNEvaluationRequest.hpp"
#include "search/NNEvaluationServiceBase.hpp"
#include "util/RecyclingAllocPool.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <functional>

namespace search {

// SimpleNNEvaluationService is a simple class that implements the NNEvaluationServiceBase
// interface. It is simple in the sense that its evaluate() method never yields. It is only
// suitable for unit-test mocking purposes, and for the UniformNNEvaluationService.
template <::alpha0::concepts::Spec Spec>
class SimpleNNEvaluationService : public search::NNEvaluationServiceBase<Spec> {
 public:
  using Game = Spec::Game;
  using InputFrame = Spec::InputFrame;
  using NetworkHeads = Spec::NetworkHeads;
  using GraphTraits = alpha0::GraphTraits<Spec>;
  using TensorEncodings = Spec::TensorEncodings;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeads>;
  using NNEvaluationRequest =
    search::NNEvaluationRequest<GraphTraits, TensorEncodings, NNEvaluation>;
  using Item = NNEvaluationRequest::Item;
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

}  // namespace search

#include "inline/search/SimpleNNEvaluationService.inl"
