#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/concepts/AuxStateConcept.hpp"
#include "search/concepts/EdgeConcept.hpp"
#include "search/concepts/ManagerParamsConcept.hpp"

namespace search {
namespace concepts {

// Everything in Traits except for Algorithms
template <class T>
concept Traits = requires {
  typename T::Game;
  typename T::EvalSpec;
  typename T::NodeStableData;
  typename T::NodeStats;
  typename T::Edge;
  typename T::ManagerParams;
  typename T::AuxState;
  typename T::SearchResults;
  typename T::TrainingInfo;
  typename T::GameLogCompactRecord;
  typename T::GameLogFullRecord;
  typename T::GameLogView;

  requires core::concepts::Game<typename T::Game>;
  requires core::concepts::EvalSpec<typename T::EvalSpec>;
  // requires search::concepts::Node<typename T::Node, typename T::EvalSpec>;
  requires search::concepts::Edge<typename T::Edge>;
  requires search::concepts::ManagerParams<typename T::ManagerParams>;
  requires search::concepts::AuxState<typename T::AuxState, typename T::ManagerParams>;
};

}  // namespace concepts
}  // namespace search
