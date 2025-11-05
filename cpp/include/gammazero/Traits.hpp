#pragma once

#include "gammazero/AuxState.hpp"
#include "gammazero/Edge.hpp"
#include "gammazero/GameLogCompactRecord.hpp"
#include "gammazero/GameLogFullRecord.hpp"
#include "gammazero/GameLogView.hpp"
#include "gammazero/ManagerParams.hpp"
#include "gammazero/NodeStableData.hpp"
#include "gammazero/NodeStats.hpp"
#include "gammazero/SearchResults.hpp"
#include "gammazero/TrainingInfo.hpp"
#include "core/Constants.hpp"
#include "core/EvalSpec.hpp"
#include "core/concepts/GameConcept.hpp"

namespace gamma0 {

template <core::concepts::Game G,
          core::concepts::EvalSpec ES = core::EvalSpec<G, core::kParadigmGammaZero>>
struct Traits {
  using Game = G;
  using EvalSpec = ES;
  using Edge = gamma0::Edge<EvalSpec>;
  using NodeStableData = gamma0::NodeStableData<EvalSpec>;
  using NodeStats = gamma0::NodeStats<EvalSpec>;
  using ManagerParams = gamma0::ManagerParams<EvalSpec>;
  using AuxState = gamma0::AuxState<ManagerParams>;
  using SearchResults = gamma0::SearchResults<Game>;
  using TrainingInfo = gamma0::TrainingInfo<Game>;
  using GameLogCompactRecord = gamma0::GameLogCompactRecord<Game>;
  using GameLogFullRecord = gamma0::GameLogFullRecord<Game>;
  using GameLogView = gamma0::GameLogView<Game>;
};

}  // namespace gamma0

// Include the binding after defining alpha0::Traits so the type is complete when
// Algorithms and concept machinery get pulled in via the binding include.
#include "gammazero/AlgorithmsBinding.hpp"  // IWYU pragma: keep
