#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/NNEvaluationRequest.hpp>

namespace mcts {

/*
 * Base class of NNEvaluationService. We pull out this base class so that we create a mock
 * implementation for testing.
 */
template <core::concepts::Game Game>
class NNEvaluationServiceBase {
 public:
  using NNEvaluationRequest = mcts::NNEvaluationRequest<Game>;

  virtual void connect() {}

  virtual void disconnect() {}

  /*
   * The key method to override. This is called by search threads to request evaluations.
   * The NNEvaluationRequest object contains a vector of NNEvaluationRequest::Item objects. Each
   * object has an eval_ member that is nullptr. The service is expected to fill in this member
   * with the result of the evaluation.
   */
  virtual void evaluate(const NNEvaluationRequest&) = 0;

  virtual void end_session() {}
};

}  // namespace mcts
