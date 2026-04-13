#pragma once

#include "core/concepts/ParadigmSpecConcept.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/concepts/AuxStateConcept.hpp"
#include "search/concepts/EdgeConcept.hpp"
#include "search/concepts/ManagerParamsConcept.hpp"

namespace search {
namespace concepts {

// Validates that a type bundle provides all required nested types for MCTS infrastructure.
template <class T>
concept Spec = requires {
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
  requires core::concepts::ParadigmSpec<typename T::EvalSpec>;
  requires search::concepts::Edge<typename T::Edge, typename T::EvalSpec>;
  requires search::concepts::ManagerParams<typename T::ManagerParams>;
  requires search::concepts::AuxState<typename T::AuxState, typename T::ManagerParams>;
};

}  // namespace concepts
}  // namespace search
