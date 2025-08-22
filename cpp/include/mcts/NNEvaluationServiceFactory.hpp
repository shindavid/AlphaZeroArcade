#pragma once

#include "core/GameServerBase.hpp"
#include "mcts/NNEvaluationServiceBase.hpp"

#include <memory>

namespace mcts {

template <typename Traits>
class NNEvaluationServiceFactory {
 public:
  using ManagerParams = Traits::ManagerParams;

  using ServiceBase = mcts::NNEvaluationServiceBase<Traits>;
  using ServiceBase_ptr = std::shared_ptr<ServiceBase>;

  // Factory method to create a service
  static ServiceBase_ptr create(const ManagerParams& params, core::GameServerBase* server);
};

}  // namespace mcts

#include "inline/mcts/NNEvaluationServiceFactory.inl"
