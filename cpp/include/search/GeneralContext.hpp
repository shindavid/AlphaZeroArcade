#pragma once

#include "core/BasicTypes.hpp"
#include "core/InputTensorizor.hpp"
#include "search/LookupTable.hpp"
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

  using Symmetries = Game::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;

  using LookupTable = search::LookupTable<Traits>;
  using InputTensorizor = core::InputTensorizor<Game>;

  struct RootInfo {
    void clear();

    InputTensorizor input_tensorizor;
    core::node_pool_index_t node_index = -1;
    core::seat_index_t active_seat = -1;
    bool add_noise = false;
  };

  GeneralContext(const ManagerParams& mparams, core::mutex_vec_sptr_t node_mutex_pool);
  void clear();
  void step();
  void jump_to(core::step_t step);

  const ManagerParams manager_params;
  const SearchParams pondering_search_params;

  AuxState aux_state;
  LookupTable lookup_table;
  RootInfo root_info;
  SearchParams search_params;
};

}  // namespace search

#include "inline/search/GeneralContext.inl"
