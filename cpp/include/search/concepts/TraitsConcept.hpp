#pragma once

#include "search/concepts/GeneralContextTraitsConcept.hpp"
#include "search/concepts/EvalRequestConcept.hpp"
#include "search/concepts/EvalResponseConcept.hpp"
#include "search/concepts/EvalServiceBaseConcept.hpp"
#include "search/concepts/EvalServiceFactoryConcept.hpp"

namespace search {
namespace concepts {

// Everything in Traits except for Algorithms
template <class T>
concept Traits = requires {
  requires search::concepts::GeneralContextTraits<T>;
  requires search::concepts::EvalRequest<typename T::EvalRequest>;
  requires search::concepts::EvalResponse<typename T::EvalResponse>;
  requires search::concepts::EvalServiceBase<typename T::EvalServiceBase, typename T::EvalRequest>;
  requires search::concepts::EvalServiceFactory<
    typename T::EvalServiceFactory, typename T::EvalServiceBase, typename T::ManagerParams>;
};

}  // namespace concepts
}  // namespace search
