#pragma once

#include "betazero/AuxState.hpp"
#include "betazero/Edge.hpp"
#include "betazero/GameLogCompactRecord.hpp"
#include "betazero/GameLogFullRecord.hpp"
#include "betazero/GameLogView.hpp"
#include "betazero/ManagerParams.hpp"
#include "betazero/Node.hpp"
#include "betazero/SearchResults.hpp"
#include "betazero/TrainingInfo.hpp"
#include "core/Constants.hpp"
#include "core/EvalSpec.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game G,
          core::concepts::EvalSpec ES = core::EvalSpec<G, core::kParadigmBetaZero>>
struct Traits {
  using Game = G;
  using EvalSpec = ES;
  using Edge = beta0::Edge;
  using Node = beta0::Node<EvalSpec>;
  using ManagerParams = beta0::ManagerParams<EvalSpec>;
  using AuxState = beta0::AuxState<ManagerParams>;
  using SearchResults = beta0::SearchResults<Game>;
  using TrainingInfo = beta0::TrainingInfo<Game>;
  using GameLogCompactRecord = beta0::GameLogCompactRecord<Game>;
  using GameLogFullRecord = beta0::GameLogFullRecord<Game>;
  using GameLogView = beta0::GameLogView<Game>;
};

}  // namespace beta0

// Include the binding after defining alpha0::Traits so the type is complete when
// Algorithms and concept machinery get pulled in via the binding include.
#include "betazero/AlgorithmsBinding.hpp"  // IWYU pragma: keep
