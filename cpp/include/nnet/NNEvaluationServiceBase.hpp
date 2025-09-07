#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "nnet/NNEvaluation.hpp"
#include "nnet/NNEvaluationRequest.hpp"

#include <memory>

namespace nnet {

/*
 * Base class of NNEvaluationService. We pull out this base class so that we create a mock
 * implementation for testing.
 *
 * In this base class, we also add free_eval() and alloc_eval() methods, which can be used in
 * place of new/delete. Doing so makes use of an underlying object pool and recycles
 * NNEvaluation objects.
 */
template <core::concepts::EvalSpec EvalSpec>
class NNEvaluationServiceBase {
 public:
  using Game = EvalSpec::Game;
  using NNEvaluation = nnet::NNEvaluation<EvalSpec>;
  using NNEvaluationRequest = nnet::NNEvaluationRequest<EvalSpec, NNEvaluation>;
  using sptr = std::shared_ptr<NNEvaluationServiceBase>;

  virtual ~NNEvaluationServiceBase() {}

  virtual void connect() {}

  virtual void disconnect() {}

  /*
   * The key method to override. This is called by search threads to request evaluations.
   * The NNEvaluationRequest object contains a vector of NNEvaluationRequest::Item objects. Each
   * object has an eval_ member that is nullptr. The service is expected to fill in this member
   * with the result of the evaluation.
   *
   * If the service needs to compute in the background, it should return core::kYield, and later
   * notify the calling thread via the request's notification_unit. Otherwise, it should return
   * core::kContinue.
   */
  virtual core::yield_instruction_t evaluate(NNEvaluationRequest& request) = 0;

  virtual void end_session() {}
};

}  // namespace nnet
