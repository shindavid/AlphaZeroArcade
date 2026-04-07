#pragma once

#include "core/BasicTypes.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

namespace search {

template <search::concepts::SearchSpec SearchSpec>
struct TrainingInfoParams {
  using SearchResults = SearchSpec::SearchResults;
  using Game = SearchSpec::Game;
  using EvalSpec = SearchSpec::EvalSpec;
  using InputEncoder = EvalSpec::TensorEncodings::InputEncoder;
  using InputFrame = EvalSpec::InputFrame;
  using Move = Game::Move;

  InputFrame frame;
  const SearchResults* mcts_results;
  Move move;
  core::seat_index_t seat;
  bool use_for_training;
  bool previous_used_for_training;
};

}  // namespace search
