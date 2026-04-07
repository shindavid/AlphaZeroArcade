#pragma once

#include "core/GameServerBase.hpp"
#include "search/NNEvaluationServiceBase.hpp"
#include "search/NNEvaluationServiceParams.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

#include <memory>

namespace search {

template <search::concepts::SearchSpec SearchSpec>
class NNEvaluationServiceFactory {
 public:
  using ServiceBase = search::NNEvaluationServiceBase<SearchSpec>;
  using ServiceBase_ptr = std::shared_ptr<ServiceBase>;

  // Factory method to create a service
  static ServiceBase_ptr create(const NNEvaluationServiceParams& params,
                                core::GameServerBase* server);
};

}  // namespace search

#include "inline/search/NNEvaluationServiceFactory.inl"
