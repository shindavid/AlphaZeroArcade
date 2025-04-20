#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationRequest.hpp>

namespace mcts {

// Return type of NNEvaluationServiceBase::evaluate().
//
//  - sequence_id: An ID assigned to the response that can then be passed to wait_for() to block
//    until the evaluation is complete.
//
//  - yield_instruction: If the service decides to handle the request asynchronously, it should set
//    this to core::kYield. This informs the requesting thread that it should yield control to
//    another thread, coming back for the result later via a wait_for() call. Otherwise, it should
//    set this to core::kContinue.
struct NNEvaluationResponse {
  NNEvaluationResponse(core::nn_evaluation_sequence_id_t s, core::yield_instruction_t y)
      : sequence_id(s), yield_instruction(y) {}

  core::nn_evaluation_sequence_id_t sequence_id;
  core::yield_instruction_t yield_instruction;
};

/*
 * Base class of NNEvaluationService. We pull out this base class so that we create a mock
 * implementation for testing.
 *
 * In this base class, we also add free_eval() and alloc_eval() methods, which can be used in
 * place of new/delete. Doing so makes use of an underlying object pool and recycles
 * NNEvaluation objects.
 */
template <core::concepts::Game Game>
class NNEvaluationServiceBase {
 public:
  using NNEvaluation = mcts::NNEvaluation<Game>;
  using NNEvaluationRequest = mcts::NNEvaluationRequest<Game>;

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
   * notify the caller via request.notify(). Otherwise, it should return core::kContinue. It is
   * assumed that request is a reference that will never go out of scope, meaning that the service
   * can store the request object and notify it later.
   */
  virtual NNEvaluationResponse evaluate(NNEvaluationRequest& request) = 0;

  // Used in conjunction with evaluate(). See NNEvaluationResponse documentation.
  virtual void wait_for(core::nn_evaluation_sequence_id_t sequence_id) = 0;

  virtual void end_session() {}
};

}  // namespace mcts
