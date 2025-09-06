#pragma once

#include "core/MctsEvalSpec.hpp"
#include "core/concepts/Game.hpp"
#include "mcts/AuxState.hpp"
#include "mcts/Edge.hpp"
#include "mcts/ManagerParams.hpp"
#include "mcts/Node.hpp"
#include "mcts/SearchResults.hpp"
#include "nnet/NNEvaluation.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"
#include "nnet/NNEvaluationServiceFactory.hpp"

namespace mcts {

template <core::concepts::Game G>
struct Traits {
  using Game = G;
  using EvalSpec = core::mcts::EvalSpec<G>;
  using Edge = mcts::Edge;
  using Node = mcts::Node<Game>;
  using ManagerParams = mcts::ManagerParams<Game>;
  using AuxState = mcts::AuxState<ManagerParams>;
  using Evaluation = nnet::NNEvaluation<EvalSpec>;
  using EvalServiceBase = nnet::NNEvaluationServiceBase<EvalSpec>;
  using EvalServiceFactory = nnet::NNEvaluationServiceFactory<EvalSpec>;
  using SearchResults = mcts::SearchResults<Game>;
};

}  // namespace mcts

// Include the binding after defining mcts::Traits so the type is complete when
// Algorithms and concept machinery get pulled in via the binding include.
#include "mcts/AlgorithmsBinding.hpp"  // IWYU pragma: keep
