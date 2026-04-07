#pragma once

#include "search/NNEvaluation.hpp"
#include "search/NNEvaluationRequest.hpp"
#include "search/SimpleNNEvaluationService.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

namespace search {

/*
 * UniformNNEvaluationService
 * A service for evaluating neural network requests with uniform probabilities.
 * This class provides a uniform evaluation service for neural network requests. It is designed to
 * support generation-0 self-play scenarios where a neural network model is not yet available.
 * The service assigns uniform probabilities to all valid actions.
 */
template <search::concepts::SearchSpec SearchSpec>
class UniformNNEvaluationService : public search::SimpleNNEvaluationService<SearchSpec> {
 public:
  using NNEvaluation = search::NNEvaluation<SearchSpec>;
  using NNEvaluationRequest = search::NNEvaluationRequest<SearchSpec>;
  using Item = NNEvaluationRequest::Item;

  UniformNNEvaluationService();
};

}  // namespace search

#include "inline/search/UniformNNEvaluationService.inl"
