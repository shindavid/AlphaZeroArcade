#pragma once

#include "alpha0/AuxState.hpp"
#include "alpha0/Edge.hpp"
#include "alpha0/GameLogCompactRecord.hpp"
#include "alpha0/GameLogFullRecord.hpp"
#include "alpha0/GameLogView.hpp"
#include "alpha0/ManagerParams.hpp"
#include "alpha0/NodeStableData.hpp"
#include "alpha0/NodeStats.hpp"
#include "alpha0/SearchResults.hpp"
#include "alpha0/TrainingInfo.hpp"
#include "alpha0/VerboseData.hpp"
#include "core/Node.hpp"
#include "alpha0/concepts/SpecConcept.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec ES>
struct Spec : ES {
  using EvalSpec = ES;
  using Edge = alpha0::Edge<ES>;
  using NodeStableData = alpha0::NodeStableData<ES>;
  using NodeStats = alpha0::NodeStats<ES>;
  using Node = core::Node<NodeStableData, NodeStats>;
  using ManagerParams = alpha0::ManagerParams<ES>;
  using AuxState = alpha0::AuxState<ManagerParams>;
  using SearchResults = alpha0::SearchResults<ES>;
  using TrainingInfo = alpha0::TrainingInfo<ES>;
  using GameLogCompactRecord = alpha0::GameLogCompactRecord<ES>;
  using GameLogFullRecord = alpha0::GameLogFullRecord<ES>;
  using GameLogView = alpha0::GameLogView<ES>;
  using VerboseData = alpha0::VerboseData<ES>;
};

}  // namespace alpha0
