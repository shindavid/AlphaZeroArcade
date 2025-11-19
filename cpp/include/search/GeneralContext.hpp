#pragma once

#include "core/BasicTypes.hpp"
#include "search/LookupTable.hpp"
#include "search/RootInfo.hpp"
#include "search/SearchParams.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/FiniteGroups.hpp"

namespace search {

// GeneralContext<Traits> contains data members that apply to the entire game tree.
template <search::concepts::Traits Traits>
struct GeneralContext {
  using Game = Traits::Game;
  using Edge = Traits::Edge;
  using ManagerParams = Traits::ManagerParams;
  using AuxState = Traits::AuxState;

  using TraitsTypes = search::TraitsTypes<Traits>;
  using Node = TraitsTypes::Node;

  using Rules = Game::Rules;
  using State = Game::State;
  using StateHistory = TraitsTypes::StateHistory;
  using Symmetries = Game::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;

  using LookupTable = search::LookupTable<Traits>;
  using RootInfo = search::RootInfo<Traits>;

  GeneralContext(const ManagerParams& mparams, core::mutex_vec_sptr_t node_mutex_pool);
  void clear();
  void step();

  const ManagerParams manager_params;
  const SearchParams pondering_search_params;

  AuxState aux_state;
  LookupTable lookup_table;
  RootInfo root_info;
  SearchParams search_params;
};

}  // namespace search

#include "inline/search/GeneralContext.inl"
