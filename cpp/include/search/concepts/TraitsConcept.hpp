#pragma once

#include "core/concepts/Game.hpp"
#include "search/concepts/AuxStateConcept.hpp"
#include "search/concepts/EdgeConcept.hpp"
#include "search/concepts/ManagerParamsConcept.hpp"
#include "search/concepts/NodeConcept.hpp"

namespace search {
namespace concepts {

template <class GT>
concept GraphTraits = requires {
  requires core::concepts::Game<typename GT::Game>;
  requires search::concepts::Node<typename GT::Node, typename GT::Game>;
  requires search::concepts::Edge<typename GT::Edge>;
};

template <class GCT>
concept GeneralContextTraits = requires {
  requires search::concepts::GraphTraits<GCT>;
  requires search::concepts::ManagerParams<typename GCT::ManagerParams>;
  requires search::concepts::AuxState<typename GCT::AuxState, typename GCT::ManagerParams>;
};

template <class T>
concept Traits = requires {
  requires search::concepts::GeneralContextTraits<T>;
  typename T::Algorithms;
  typename T::EvalRequest;
  typename T::EvalResponse;
  typename T::EvalServiceBase;
  typename T::EvalServiceFactory;
};

}  // namespace concepts
}  // namespace search
