#pragma once

#include "core/BasicTypes.hpp"
#include "core/InputTensorizor.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace search {

template <search::concepts::Traits Traits>
struct TrainingInfoParams {
  using SearchResults = Traits::SearchResults;
  using Game = Traits::Game;
  using InputTensorizor = core::InputTensorizor<Game>;
  using TensorizationUnit = InputTensorizor::Unit;

  TensorizationUnit position;
  const SearchResults* mcts_results;
  core::action_t action;
  core::action_mode_t action_mode;
  core::seat_index_t seat;
  bool use_for_training;
  bool previous_used_for_training;
};

}  // namespace search
