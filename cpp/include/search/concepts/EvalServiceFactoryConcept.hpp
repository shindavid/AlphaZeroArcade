#pragma once

#include "core/GameServerBase.hpp"

#include <concepts>
#include <memory>

namespace search {
namespace concepts {

template <class E, class EvalServiceBase, class ManagerParams>
concept EvalServiceFactory =
  requires(E& factory, const ManagerParams& params, core::GameServerBase* server) {
    { factory.create(params, server) } -> std::same_as<std::shared_ptr<EvalServiceBase>>;
  };

}  // namespace concepts
}  // namespace search
