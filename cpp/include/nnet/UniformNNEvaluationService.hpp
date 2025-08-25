#pragma once

#include "nnet/NNEvaluation.hpp"
#include "nnet/NNEvaluationRequest.hpp"
#include "nnet/SimpleNNEvaluationService.hpp"

namespace nnet {

/*
 * UniformNNEvaluationService
 * A service for evaluating neural network requests with uniform probabilities.
 * This class provides a uniform evaluation service for neural network requests. It is designed to
 * support generation-0 self-play scenarios where a neural network model is not yet available.
 * The service assigns uniform probabilities to all valid actions.
 */
template <typename Traits>
class UniformNNEvaluationService : public nnet::SimpleNNEvaluationService<Traits> {
 public:
  using NNEvaluation = nnet::NNEvaluation<Traits>;
  using NNEvaluationRequest = nnet::NNEvaluationRequest<Traits>;
  using Item = NNEvaluationRequest::Item;

  UniformNNEvaluationService();
};

}  // namespace nnet

#include "inline/nnet/UniformNNEvaluationService.inl"
