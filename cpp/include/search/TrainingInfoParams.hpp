#pragma once

#include "core/BasicTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace search {

template <search::concepts::Traits Traits>
struct TrainingInfoParams {
  using SearchResults = Traits::SearchResults;
  using Game = Traits::Game;
  using EvalSpec = Traits::EvalSpec;
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
