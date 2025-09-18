#pragma once

#include "core/GameServerBase.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"
#include "nnet/NNEvaluationServiceParams.hpp"

#include <memory>

namespace nnet {

template <search::concepts::Traits Traits>
class NNEvaluationServiceFactory {
 public:
  using ServiceBase = nnet::NNEvaluationServiceBase<Traits>;
  using ServiceBase_ptr = std::shared_ptr<ServiceBase>;

  // Factory method to create a service
  static ServiceBase_ptr create(const NNEvaluationServiceParams& params,
                                core::GameServerBase* server);
};

}  // namespace nnet

#include "inline/nnet/NNEvaluationServiceFactory.inl"
