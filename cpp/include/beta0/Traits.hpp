#pragma once

#include "beta0/AuxData.hpp"
#include "beta0/AuxState.hpp"
#include "beta0/Edge.hpp"
#include "beta0/GameLogCompactRecord.hpp"
#include "beta0/GameLogFullRecord.hpp"
#include "beta0/GameLogView.hpp"
#include "beta0/ManagerParams.hpp"
#include "beta0/NodeStableData.hpp"
#include "beta0/NodeStats.hpp"
#include "beta0/SearchResults.hpp"
#include "beta0/TrainingInfo.hpp"
#include "beta0/VerboseData.hpp"
#include "core/Constants.hpp"
#include "core/EvalSpec.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game G,
          core::concepts::EvalSpec ES = core::EvalSpec<G, core::kParadigmBetaZero>>
struct Traits {
  using Game = G;
  using EvalSpec = ES;
  using Edge = beta0::Edge<EvalSpec>;
  using NodeStableData = beta0::NodeStableData<EvalSpec>;
  using NodeStats = beta0::NodeStats<EvalSpec>;
  using ManagerParams = beta0::ManagerParams<EvalSpec>;
  using AuxState = beta0::AuxState<ManagerParams>;
  using SearchResults = beta0::SearchResults<Game>;
  using TrainingInfo = beta0::TrainingInfo<Game>;
  using GameLogCompactRecord = beta0::GameLogCompactRecord<Game>;
  using GameLogFullRecord = beta0::GameLogFullRecord<Game>;
  using GameLogView = beta0::GameLogView<Game>;
  using AuxData = beta0::AuxData<Game>;
  using VerboseData = beta0::VerboseData<Game>;
};

}  // namespace beta0

// Include the binding after defining beta0::Traits so the type is complete when
// Algorithms and concept machinery get pulled in via the binding include.
#include "beta0/AlgorithmsBinding.hpp"  // IWYU pragma: keep
