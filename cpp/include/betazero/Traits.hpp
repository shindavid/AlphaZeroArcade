#pragma once

#include "alphazero/AuxState.hpp"
#include "alphazero/Edge.hpp"
#include "alphazero/ManagerParams.hpp"
#include "alphazero/Node.hpp"
#include "alphazero/SearchResults.hpp"
#include "core/Constants.hpp"
#include "core/EvalSpec.hpp"
#include "core/concepts/GameConcept.hpp"
#include "nnet/NNEvaluation.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"
#include "nnet/NNEvaluationServiceFactory.hpp"

namespace beta0 {

// For now, beta0::Traits uses the same classes as alpha0::Traits. Later we will specialize it.
template <core::concepts::Game G>
struct Traits {
  using Game = G;
  using EvalSpec = core::EvalSpec<Game, core::kParadigmBetaZero>;
  using Edge = alpha0::Edge;
  using Node = alpha0::Node<EvalSpec>;
  using ManagerParams = alpha0::ManagerParams<EvalSpec>;
  using AuxState = alpha0::AuxState<ManagerParams>;
  using Evaluation = nnet::NNEvaluation<EvalSpec>;
  using EvalServiceBase = nnet::NNEvaluationServiceBase<EvalSpec>;
  using EvalServiceFactory = nnet::NNEvaluationServiceFactory<EvalSpec>;
  using SearchResults = alpha0::SearchResults<Game>;
};

}  // namespace beta0

// Include the binding after defining alpha0::Traits so the type is complete when
// Algorithms and concept machinery get pulled in via the binding include.
#include "betazero/AlgorithmsBinding.hpp"  // IWYU pragma: keep
