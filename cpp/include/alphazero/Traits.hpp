#pragma once

#include "alphazero/AuxState.hpp"
#include "alphazero/Edge.hpp"
#include "alphazero/GameLogCompactRecord.hpp"
#include "alphazero/GameLogFullRecord.hpp"
#include "alphazero/GameLogView.hpp"
#include "alphazero/ManagerParams.hpp"
#include "alphazero/NodeStableData.hpp"
#include "alphazero/NodeStats.hpp"
#include "alphazero/SearchResults.hpp"
#include "alphazero/TrainingInfo.hpp"
#include "core/Constants.hpp"
#include "core/EvalSpec.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "core/concepts/GameConcept.hpp"

namespace alpha0 {

template <core::concepts::Game G,
          core::concepts::EvalSpec ES = core::EvalSpec<G, core::kParadigmAlphaZero>>
struct Traits {
  using Game = G;
  using EvalSpec = ES;
  using Edge = alpha0::Edge;
  using NodeStableData = alpha0::NodeStableData<EvalSpec>;
  using NodeStats = alpha0::NodeStats<EvalSpec>;
  using ManagerParams = alpha0::ManagerParams<EvalSpec>;
  using AuxState = alpha0::AuxState<ManagerParams>;
  using SearchResults = alpha0::SearchResults<Game>;
  using TrainingInfo = alpha0::TrainingInfo<Game>;
  using GameLogCompactRecord = alpha0::GameLogCompactRecord<Game>;
  using GameLogFullRecord = alpha0::GameLogFullRecord<Game>;
  using GameLogView = alpha0::GameLogView<Game>;
};

}  // namespace alpha0

// Include the binding after defining alpha0::Traits so the type is complete when
// Algorithms and concept machinery get pulled in via the binding include.
#include "alphazero/AlgorithmsBinding.hpp"  // IWYU pragma: keep
