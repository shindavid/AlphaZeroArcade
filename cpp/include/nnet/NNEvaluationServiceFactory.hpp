#pragma once

#include "core/GameServerBase.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"
#include "nnet/NNEvaluationServiceParams.hpp"

#include <memory>

namespace nnet {

template <core::concepts::EvalSpec EvalSpec>
class NNEvaluationServiceFactory {
 public:
  using ServiceBase = nnet::NNEvaluationServiceBase<EvalSpec>;
  using ServiceBase_ptr = std::shared_ptr<ServiceBase>;

  // Factory method to create a service
  static ServiceBase_ptr create(const NNEvaluationServiceParams& params,
                                core::GameServerBase* server);
};

}  // namespace nnet

#include "inline/nnet/NNEvaluationServiceFactory.inl"
