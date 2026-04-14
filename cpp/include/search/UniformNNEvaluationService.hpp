#pragma once

#include "alpha0/GraphTraits.hpp"
#include "alpha0/concepts/SpecConcept.hpp"
#include "search/NNEvaluation.hpp"
#include "search/NNEvaluationRequest.hpp"
#include "search/SimpleNNEvaluationService.hpp"

namespace search {

/*
 * UniformNNEvaluationService
 * A service for evaluating neural network requests with uniform probabilities.
 * This class provides a uniform evaluation service for neural network requests. It is designed to
 * support generation-0 self-play scenarios where a neural network model is not yet available.
 * The service assigns uniform probabilities to all valid actions.
 */
template <::alpha0::concepts::Spec Spec>
class UniformNNEvaluationService : public search::SimpleNNEvaluationService<Spec> {
 public:
  using Game = Spec::Game;
  using InputFrame = Spec::InputFrame;
  using NetworkHeadsList = Spec::NetworkHeads::List;
  using GraphTraits = alpha0::GraphTraits<Spec>;
  using TensorEncodings = Spec::TensorEncodings;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeadsList>;
  using NNEvaluationRequest =
    search::NNEvaluationRequest<GraphTraits, TensorEncodings, NNEvaluation>;
  using Item = NNEvaluationRequest::Item;

  UniformNNEvaluationService();
};

}  // namespace search

#include "inline/search/UniformNNEvaluationService.inl"
