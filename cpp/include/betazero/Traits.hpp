#pragma once

#include "betazero/AuxState.hpp"
#include "betazero/Edge.hpp"
#include "betazero/Node.hpp"
#include "betazero/ManagerParams.hpp"
#include "betazero/SearchResults.hpp"
#include "core/Constants.hpp"
#include "core/EvalSpec.hpp"
#include "core/concepts/GameConcept.hpp"
#include "nnet/NNEvaluation.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"
#include "nnet/NNEvaluationServiceFactory.hpp"

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
  using Evaluation = nnet::NNEvaluation<EvalSpec>;
  using EvalServiceBase = nnet::NNEvaluationServiceBase<EvalSpec>;
  using EvalServiceFactory = nnet::NNEvaluationServiceFactory<EvalSpec>;
  using SearchResults = beta0::SearchResults<Game>;
};

}  // namespace beta0

// Include the binding after defining alpha0::Traits so the type is complete when
// Algorithms and concept machinery get pulled in via the binding include.
#include "betazero/AlgorithmsBinding.hpp"  // IWYU pragma: keep
