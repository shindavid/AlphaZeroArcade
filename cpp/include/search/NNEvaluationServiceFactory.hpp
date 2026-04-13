#pragma once

#include "core/GameServerBase.hpp"
#include "search/NNEvaluationServiceBase.hpp"
#include "search/NNEvaluationServiceParams.hpp"
#include "search/concepts/SpecConcept.hpp"

#include <memory>

namespace search {

template <search::concepts::Spec Spec>
class NNEvaluationServiceFactory {
 public:
  using ServiceBase = search::NNEvaluationServiceBase<Spec>;
  using ServiceBase_ptr = std::shared_ptr<ServiceBase>;

  // Factory method to create a service
  static ServiceBase_ptr create(const NNEvaluationServiceParams& params,
                                core::GameServerBase* server);
};

}  // namespace search

#include "inline/search/NNEvaluationServiceFactory.inl"
