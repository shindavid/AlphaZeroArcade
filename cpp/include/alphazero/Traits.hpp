#pragma once

#include "core/MctsEvalSpec.hpp"
#include "core/concepts/Game.hpp"
#include "a0/AuxState.hpp"
#include "a0/Edge.hpp"
#include "a0/ManagerParams.hpp"
#include "a0/Node.hpp"
#include "a0/SearchResults.hpp"
#include "nnet/NNEvaluation.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"
#include "nnet/NNEvaluationServiceFactory.hpp"

namespace a0 {

template <core::concepts::Game G>
struct Traits {
  using Game = G;
  using EvalSpec = core::a0::EvalSpec<G>;
  using Edge = a0::Edge;
  using Node = a0::Node<Game>;
  using ManagerParams = a0::ManagerParams<Game>;
  using AuxState = a0::AuxState<ManagerParams>;
  using Evaluation = nnet::NNEvaluation<EvalSpec>;
  using EvalServiceBase = nnet::NNEvaluationServiceBase<EvalSpec>;
  using EvalServiceFactory = nnet::NNEvaluationServiceFactory<EvalSpec>;
  using SearchResults = a0::SearchResults<Game>;
};

}  // namespace a0

// Include the binding after defining a0::Traits so the type is complete when
// Algorithms and concept machinery get pulled in via the binding include.
#include "a0/AlgorithmsBinding.hpp"  // IWYU pragma: keep
