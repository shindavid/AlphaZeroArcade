#pragma once

#include "alphazero/AuxState.hpp"
#include "alphazero/Edge.hpp"
#include "alphazero/ManagerParams.hpp"
#include "alphazero/Node.hpp"
#include "alphazero/SearchResults.hpp"
#include "core/Constants.hpp"
#include "core/EvalSpec.hpp"
#include "core/concepts/Game.hpp"
#include "nnet/NNEvaluation.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"
#include "nnet/NNEvaluationServiceFactory.hpp"

namespace alpha0 {

template <core::concepts::Game G>
struct Traits {
  using Game = G;
  using EvalSpec = core::EvalSpec<Game, core::kParadigmAlphaZero>;
  using Edge = alpha0::Edge;
  using Node = alpha0::Node<Game>;
  using ManagerParams = alpha0::ManagerParams<Game>;
  using AuxState = alpha0::AuxState<ManagerParams>;
  using Evaluation = nnet::NNEvaluation<EvalSpec>;
  using EvalServiceBase = nnet::NNEvaluationServiceBase<EvalSpec>;
  using EvalServiceFactory = nnet::NNEvaluationServiceFactory<EvalSpec>;
  using SearchResults = alpha0::SearchResults<Game>;
};

}  // namespace alpha0

// Include the binding after defining alpha0::Traits so the type is complete when
// Algorithms and concept machinery get pulled in via the binding include.
#include "alphazero/AlgorithmsBinding.hpp"  // IWYU pragma: keep
