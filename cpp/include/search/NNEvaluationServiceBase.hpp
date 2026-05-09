#pragma once

#include "core/AuxEvalService.hpp"
#include "core/BasicTypes.hpp"
#include "core/ModelBundle.hpp"
#include "search/NNEvalTraits.hpp"
#include "search/NNEvaluationRequest.hpp"

#include <functional>
#include <memory>
#include <utility>

namespace search {

/*
 * Base class of NNEvaluationService. We pull out this base class so that we create a mock
 * implementation for testing.
 *
 * In this base class, we also add free_eval() and alloc_eval() methods, which can be used in
 * place of new/delete. Doing so makes use of an underlying object pool and recycles
 * NNEvaluation objects.
 *
 * The base also owns an optional core::AuxEvalService — a paradigm-specific auxiliary
 * evaluator (e.g. beta0::BackupNNEvaluator) constructed via an AuxFactory callback at
 * service creation time. See core/AuxEvalService.hpp for the contract.
 */
template <search::concepts::NNEvalTraits Traits>
class NNEvaluationServiceBase {
 public:
  using NNEvaluation = Traits::NNEvaluation;
  using NNEvaluationRequest = search::NNEvaluationRequest<Traits>;
  using sptr = std::shared_ptr<NNEvaluationServiceBase>;

  // Callback used to construct the auxiliary evaluation service. Invoked at most once per
  // NNEvaluationService instance. The returned aux is default-constructed (unloaded);
  // NNEvaluationService is responsible for driving its reload_weights() at construction
  // (if a local model file was provided) and on every subsequent wire-pushed reload.
  using AuxFactory = std::function<std::unique_ptr<core::AuxEvalService>()>;

  virtual ~NNEvaluationServiceBase() {}

  // Returns the auxiliary evaluation service associated with this NN service, or nullptr
  // if none was configured. Ownership remains with this service.
  core::AuxEvalService* aux_service() const { return aux_service_.get(); }

  // Setter intended for test fixtures that construct a SimpleNNEvaluationService subclass
  // directly (production code installs the aux service via the AuxFactory wired into the
  // NNEvaluationService constructor).
  void set_aux_service(std::unique_ptr<core::AuxEvalService> aux) {
    aux_service_ = std::move(aux);
  }

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

 protected:
  std::unique_ptr<core::AuxEvalService> aux_service_;
};

}  // namespace search
