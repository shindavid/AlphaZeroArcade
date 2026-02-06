#pragma once

#include "core/BasicTypes.hpp"
#include "core/InputTensorizor.hpp"
#include "core/StateIterator.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace search {

// GeneralContext<Traits> contains data members that apply to the entire game tree.
template <search::concepts::Traits Traits>
struct GeneralContext {
  using Game = Traits::Game;
  using ManagerParams = Traits::ManagerParams;
  using AuxState = Traits::AuxState;

  using Rules = Game::Rules;
  using State = Game::State;

  using LookupTable = search::LookupTable<Traits>;
  using InputTensorizor = core::InputTensorizor<Game>;
  using StateIterator = core::StateIterator<Game>;

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
  void jump_to(StateIterator it, core::step_t step);

  const ManagerParams manager_params;
  const SearchParams pondering_search_params;

  AuxState aux_state;
  LookupTable lookup_table;
  RootInfo root_info;
  SearchParams search_params;
};

}  // namespace search

#include "inline/search/GeneralContext.inl"
