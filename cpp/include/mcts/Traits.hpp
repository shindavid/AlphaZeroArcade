#pragma once

#include "core/concepts/Game.hpp"
#include "mcts/Algorithms.hpp"
#include "mcts/AuxState.hpp"
#include "mcts/Edge.hpp"
#include "mcts/ManagerParams.hpp"
#include "mcts/Node.hpp"
#include "mcts/SearchResults.hpp"
#include "nnet/NNEvaluation.hpp"
#include "nnet/NNEvaluationRequest.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"
#include "nnet/NNEvaluationServiceFactory.hpp"
#include "search/concepts/Traits.hpp"

namespace mcts {

template <core::concepts::Game G>
struct Traits {
  using Game = G;
  using Node = mcts::Node<Traits>;
  using Edge = mcts::Edge;
  using AuxState = mcts::AuxState<Traits>;
  using Algorithms = mcts::Algorithms<Traits>;
  using ManagerParams = mcts::ManagerParams<Game>;
  using EvalRequest = nnet::NNEvaluationRequest<Game>;
  using EvalResponse = nnet::NNEvaluation<Game>;
  using EvalServiceBase = nnet::NNEvaluationServiceBase<Game>;
  using EvalServiceFactory = nnet::NNEvaluationServiceFactory<Game>;
  using SearchResults = mcts::SearchResults<Game>;

  static_assert(search::concepts::Traits<Traits>);
};

}  // namespace mcts
