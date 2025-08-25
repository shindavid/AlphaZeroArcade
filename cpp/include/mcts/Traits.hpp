#pragma once

#include "core/concepts/Game.hpp"
#include "mcts/Algorithms.hpp"
#include "mcts/AuxState.hpp"
#include "mcts/Edge.hpp"
#include "mcts/Node.hpp"
#include "mcts/SearchResults.hpp"
#include "nnet/NNEvaluation.hpp"
#include "nnet/NNEvaluationRequest.hpp"
#include "nnet/NNEvaluationServiceBase.hpp"
#include "nnet/NNEvaluationServiceFactory.hpp"
#include "search/ManagerParams.hpp"
#include "search/concepts/Traits.hpp"

namespace mcts {

template <core::concepts::Game G>
struct Traits {
  using Game = G;
  using Node = mcts::Node<Traits>;
  using Edge = mcts::Edge;
  using AuxState = mcts::AuxState<Traits>;
  using Algorithms = mcts::Algorithms<Traits>;
  using ManagerParams = search::ManagerParams<Traits>;
  using EvalRequest = nnet::NNEvaluationRequest<Traits>;
  using EvalResponse = nnet::NNEvaluation<Traits>;
  using EvalServiceBase = nnet::NNEvaluationServiceBase<Traits>;
  using EvalServiceFactory = nnet::NNEvaluationServiceFactory<Traits>;
  using SearchResults = mcts::SearchResults<Traits>;

  static_assert(search::concepts::Traits<Traits>);
};

}  // namespace mcts
