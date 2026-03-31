#pragma once

#include "core/BasicTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace search {

template <search::concepts::Traits Traits>
struct TrainingInfoParams {
  using SearchResults = Traits::SearchResults;
  using Game = Traits::Game;
  using EvalSpec = Traits::EvalSpec;
  using InputTensorizor = EvalSpec::InputTensorizor;
  using InputFrame = EvalSpec::InputFrame;
  using Move = Game::Move;

  InputFrame frame;
  const SearchResults* mcts_results;
  Move move;
  core::game_phase_t game_phase;
  core::seat_index_t seat;
  bool use_for_training;
  bool previous_used_for_training;
};

}  // namespace search
