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
#include "core/EvalSpec.hpp"
#include "core/Node.hpp"
#include "core/SearchParadigm.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "core/concepts/GameConcept.hpp"

namespace alpha0 {

template <core::concepts::Game G,
          core::concepts::EvalSpec ES = core::EvalSpec<G, core::kParadigmAlphaZero>>
struct SearchSpec {
  using Game = G;
  using EvalSpec = ES;
  using Edge = alpha0::Edge<EvalSpec>;
  using NodeStableData = alpha0::NodeStableData<EvalSpec>;
  using NodeStats = alpha0::NodeStats<EvalSpec>;
  using Node = core::Node<NodeStableData, NodeStats>;
  using ManagerParams = alpha0::ManagerParams<EvalSpec>;
  using AuxState = alpha0::AuxState<ManagerParams>;
  using SearchResults = alpha0::SearchResults<EvalSpec>;
  using TrainingInfo = alpha0::TrainingInfo<EvalSpec>;
  using GameLogCompactRecord = alpha0::GameLogCompactRecord<EvalSpec>;
  using GameLogFullRecord = alpha0::GameLogFullRecord<EvalSpec>;
  using GameLogView = alpha0::GameLogView<EvalSpec>;
  using VerboseData = alpha0::VerboseData<EvalSpec>;
};

}  // namespace alpha0
