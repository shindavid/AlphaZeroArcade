#pragma once

#include "core/concepts/Game.hpp"
#include "search/concepts/GraphTraitsConcept.hpp"
#include "search/concepts/ManagerParamsConcept.hpp"
#include "search/concepts/AuxStateConcept.hpp"

namespace search {
namespace concepts {

template <class GCT>
concept GeneralContextTraits = requires {
  requires search::concepts::GraphTraits<GCT>;
  requires search::concepts::ManagerParams<typename GCT::ManagerParams>;
  requires search::concepts::AuxState<typename GCT::AuxState, typename GCT::ManagerParams>;
};

template <class T>
concept Traits = requires {
  requires core::concepts::Game<typename T::Game>;
  typename T::Node;
  typename T::Edge;
  typename T::AuxState;
  typename T::ManagerParams;
  typename T::Algorithms;
  typename T::EvalRequest;
  typename T::EvalResponse;
  typename T::EvalServiceBase;
  typename T::EvalServiceFactory;
};

}  // namespace concepts
}  // namespace search
