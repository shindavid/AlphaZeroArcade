#pragma once

#include "core/GameServerBase.hpp"
#include "search/NNEvalTraits.hpp"
#include "search/NNEvaluationServiceBase.hpp"
#include "search/NNEvaluationServiceParams.hpp"

#include <memory>

namespace search {

template <search::concepts::NNEvalTraits Traits>
class NNEvaluationServiceFactory {
 public:
  using ServiceBase = search::NNEvaluationServiceBase<Traits>;
  using ServiceBase_ptr = std::shared_ptr<ServiceBase>;
  using AuxFactory = ServiceBase::AuxFactory;

  // Factory method to create a service. The aux_factory callback (if non-null) is forwarded
  // to the underlying NNEvaluationService and used to construct an auxiliary evaluator
  // attached to the service. See core/AuxEvalService.hpp for the contract.
  static ServiceBase_ptr create(const NNEvaluationServiceParams& params,
                                core::GameServerBase* server,
                                AuxFactory aux_factory = nullptr);
};

}  // namespace search

#include "inline/search/NNEvaluationServiceFactory.inl"
