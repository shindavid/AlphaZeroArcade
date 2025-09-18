#pragma once

#include "core/GameServerBase.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "search/NNEvaluationServiceBase.hpp"
#include "search/NNEvaluationServiceParams.hpp"

#include <memory>

namespace search {

template <search::concepts::Traits Traits>
class NNEvaluationServiceFactory {
 public:
  using ServiceBase = search::NNEvaluationServiceBase<Traits>;
  using ServiceBase_ptr = std::shared_ptr<ServiceBase>;

  // Factory method to create a service
  static ServiceBase_ptr create(const NNEvaluationServiceParams& params,
                                core::GameServerBase* server);
};

}  // namespace search

#include "inline/search/NNEvaluationServiceFactory.inl"
