#pragma once

#include "core/GameServerBase.hpp"

#include <concepts>
#include <memory>

namespace search {
namespace concepts {

template <class E, class SearchSpec>
concept EvalServiceFactory = requires(E& factory, const typename SearchSpec::ManagerParams& params,
                                      core::GameServerBase* server) {
  {
    factory.create(params, server)
  } -> std::same_as<std::shared_ptr<typename SearchSpec::EvalServiceBase>>;
};

}  // namespace concepts
}  // namespace search
