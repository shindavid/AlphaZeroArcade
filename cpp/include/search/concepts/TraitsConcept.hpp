#pragma once

#include "nnet/NNEvaluationRequest.hpp"
#include "search/concepts/EvaluationConcept.hpp"
#include "search/concepts/EvalServiceBaseConcept.hpp"
#include "search/concepts/EvalServiceFactoryConcept.hpp"
#include "search/concepts/GeneralContextTraitsConcept.hpp"

namespace search {
namespace concepts {

// Everything in Traits except for Algorithms
template <class T>
concept Traits = requires {
  requires search::concepts::GeneralContextTraits<T>;
  requires search::concepts::Evaluation<typename T::Evaluation>;

  requires search::concepts::EvalServiceBase<
    typename T::EvalServiceBase,
    nnet::NNEvaluationRequest<typename T::EvalSpec, typename T::Evaluation>>;

  requires search::concepts::EvalServiceFactory<
    typename T::EvalServiceFactory, typename T::EvalServiceBase, typename T::ManagerParams>;
};

}  // namespace concepts
}  // namespace search
