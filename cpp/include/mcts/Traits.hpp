#pragma once

#include "core/concepts/Game.hpp"
#include "mcts/Algorithms.hpp"
#include "mcts/Edge.hpp"
#include "search/ManagerParams.hpp"
#include "mcts/NNEvaluation.hpp"
#include "mcts/NNEvaluationRequest.hpp"
#include "mcts/NNEvaluationServiceBase.hpp"
#include "mcts/NNEvaluationServiceFactory.hpp"
#include "mcts/Node.hpp"
#include "mcts/SearchResults.hpp"
#include "search/concepts/Traits.hpp"

namespace mcts {

template <core::concepts::Game G>
struct Traits {
  using Game = G;
  using Node = mcts::Node<Traits>;
  using Edge = mcts::Edge;
  using Algorithms = mcts::Algorithms<Traits>;
  using ManagerParams = mcts::ManagerParams<Traits>;
  using EvalRequest = mcts::NNEvaluationRequest<Traits>;
  using EvalResponse = mcts::NNEvaluation<Traits>;
  using EvalServiceBase = mcts::NNEvaluationServiceBase<Traits>;
  using EvalServiceFactory = mcts::NNEvaluationServiceFactory<Traits>;
  using SearchResults = mcts::SearchResults<Traits>;

  static_assert(search::concepts::Traits<Traits>);
};

}  // namespace mcts
