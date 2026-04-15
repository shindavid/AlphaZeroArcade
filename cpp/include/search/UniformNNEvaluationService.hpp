#pragma once

#include "search/NNEvalTraits.hpp"
#include "search/SimpleNNEvaluationService.hpp"

namespace search {

/*
 * UniformNNEvaluationService
 * A service for evaluating neural network requests with uniform probabilities.
 * This class provides a uniform evaluation service for neural network requests. It is designed to
 * support generation-0 self-play scenarios where a neural network model is not yet available.
 * The service assigns uniform probabilities to all valid actions.
 */
template <search::concepts::NNEvalTraits Traits>
class UniformNNEvaluationService : public search::SimpleNNEvaluationService<Traits> {
 public:
  using NNEvaluation = Traits::NNEvaluation;
  using NNEvaluationRequest = search::NNEvaluationRequest<Traits>;
  using Item = NNEvaluationRequest::Item;

  UniformNNEvaluationService();
};

}  // namespace search

#include "inline/search/UniformNNEvaluationService.inl"
