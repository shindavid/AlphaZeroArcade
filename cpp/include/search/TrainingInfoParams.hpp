#pragma once

#include "core/BasicTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace search {

template <search::concepts::Traits Traits>
struct TrainingInfoParams {
  using SearchResults = Traits::SearchResults;
  using Game = Traits::Game;
  using State = Game::State;

  State state;
  const SearchResults* mcts_results;
  core::action_t action;
  core::seat_index_t seat;
  bool use_for_training;
  bool previous_used_for_training;
};

}  // namespace search
