#pragma once

#include "core/Constants.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace search {

template <search::concepts::Traits Traits>
struct RootInfoBase {
  using StateHistory = search::TraitsTypes<Traits>::StateHistory;
  using Game = Traits::Game;
  using State = Game::State;

  void clear();

  StateHistory history;
  core::node_pool_index_t node_index = -1;
  core::seat_index_t active_seat = -1;
  bool add_noise = false;
};

template <search::concepts::Traits Traits, core::TranspositionRule TranspositionRule>
struct RootInfoImpl : public RootInfoBase<Traits> {};

template <search::concepts::Traits Traits>
struct RootInfoImpl<Traits, core::kSymmetryTranspositions> : public RootInfoBase<Traits> {
  using Base = RootInfoBase<Traits>;
  using Game = Base::Game;

  void clear();

  group::element_t canonical_sym = -1;
};

template <search::concepts::Traits Traits>
using RootInfo = RootInfoImpl<Traits, Traits::EvalSpec::MctsConfiguration::kTranspositionRule>;

}  // namespace search

#include "inline/search/RootInfo.inl"
