#pragma once

#include "core/GameServerBase.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"

#include <memory>

namespace nnet {

template <typename Traits>
class NNEvaluationServiceFactory {
 public:
  using ManagerParams = Traits::ManagerParams;

  using ServiceBase = nnet::NNEvaluationServiceBase<Traits>;
  using ServiceBase_ptr = std::shared_ptr<ServiceBase>;

  // Factory method to create a service
  static ServiceBase_ptr create(const ManagerParams& params, core::GameServerBase* server);
};

}  // namespace nnet

#include "inline/nnet/NNEvaluationServiceFactory.inl"
