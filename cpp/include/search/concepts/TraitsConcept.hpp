#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/concepts/AuxStateConcept.hpp"
#include "search/concepts/EdgeConcept.hpp"
#include "search/concepts/EvalServiceBaseConcept.hpp"
#include "search/concepts/EvalServiceFactoryConcept.hpp"
#include "search/concepts/EvaluationConcept.hpp"
#include "search/concepts/ManagerParamsConcept.hpp"
#include "search/concepts/NodeConcept.hpp"

namespace search {
namespace concepts {

// Everything in Traits except for Algorithms
template <class T>
concept Traits = requires {
  typename T::Game;
  typename T::EvalSpec;
  typename T::Node;
  typename T::Edge;
  typename T::ManagerParams;
  typename T::AuxState;
  typename T::Evaluation;
  typename T::EvalServiceBase;
  typename T::EvalServiceFactory;

  requires core::concepts::Game<typename T::Game>;
  requires core::concepts::EvalSpec<typename T::EvalSpec>;
  requires search::concepts::Node<typename T::Node, typename T::EvalSpec>;
  requires search::concepts::Edge<typename T::Edge>;
  requires search::concepts::ManagerParams<typename T::ManagerParams>;
  requires search::concepts::AuxState<typename T::AuxState, typename T::ManagerParams>;
  requires search::concepts::Evaluation<typename T::Evaluation>;
  requires search::concepts::EvalServiceBase<typename T::EvalServiceBase, T>;
  requires search::concepts::EvalServiceFactory<typename T::EvalServiceFactory, T>;
};

}  // namespace concepts
}  // namespace search
