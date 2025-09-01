#pragma once

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
  using Edge = mcts::Edge;
  using Node = mcts::Node<Game>;
  using ManagerParams = mcts::ManagerParams<Game>;
  using AuxState = mcts::AuxState<ManagerParams>;
  using Evaluation = nnet::NNEvaluation<Game>;
  using EvalServiceBase = nnet::NNEvaluationServiceBase<Game>;
  using EvalServiceFactory = nnet::NNEvaluationServiceFactory<Game>;
  using SearchResults = mcts::SearchResults<Game>;
};

}  // namespace mcts

// Include the binding after defining mcts::Traits so the type is complete when
// Algorithms and concept machinery get pulled in via the binding include.
#include "mcts/AlgorithmsBinding.hpp"  // IWYU pragma: keep
