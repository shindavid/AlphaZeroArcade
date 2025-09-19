#pragma once

#include "search/GameLogBase.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/FiniteGroups.hpp"

namespace search {

template <search::concepts::Traits Traits>
struct GameLogViewParams {
  using Game = Traits::Game;
  using GameLogBase = search::GameLogBase<Traits>;
  using GameLogCompactRecord = GameLogBase::GameLogCompactRecord;

  const GameLogCompactRecord* record = nullptr;
  const GameLogCompactRecord* next_record = nullptr;
  const Game::State* cur_pos = nullptr;
  const Game::State* final_pos = nullptr;
  const Game::Types::ValueTensor* outcome = nullptr;
  group::element_t sym = group::kIdentity;
};

}  // namespace search
