#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationRequest.hpp>
#include <mcts/NNEvaluationServiceBase.hpp>

namespace mcts{

/*
 * UniformNNEvaluationService
 * A service for evaluating neural network requests with uniform probabilities.
 * This class provides a uniform evaluation service for neural network requests. It is designed to
 * support generation-0 self-play scenarios where a neural network model is not yet available.
 * The service assigns uniform probabilities to all valid actions.
 */
template <core::concepts::Game Game>
class UniformNNEvaluationService : public mcts::NNEvaluationServiceBase<Game> {
 public:
  using NNEvaluation = mcts::NNEvaluation<Game>;
  using NNEvaluationRequest = mcts::NNEvaluationRequest<Game>;
  using ValueTensor = NNEvaluation::ValueTensor;
  using PolicyTensor = NNEvaluation::PolicyTensor;
  using ActionValueTensor = NNEvaluation::ActionValueTensor;
  using ActionMask = NNEvaluation::ActionMask;

  void evaluate(const NNEvaluationRequest&) override;
};

}  // namespace mcts

#include <inline/mcts/UniformNNEvaluationService.inl>