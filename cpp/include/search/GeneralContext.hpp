#pragma once

#include "core/BasicTypes.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"
#include "search/TypeDefs.hpp"
#include "util/FiniteGroups.hpp"

#include <array>

namespace search {

// GeneralContext<Traits> contains data members that apply to the entire game tree.
template <typename Traits>
struct GeneralContext {
  using Game = Traits::Game;
  using ManagerParams = Traits::ManagerParams;

  using Rules = Game::Rules;
  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using Symmetries = Game::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;

  using StateHistoryArray = std::array<StateHistory, SymmetryGroup::kOrder>;
  using LookupTable = search::LookupTable<Traits>;

  struct RootInfo {
    void clear();

    StateHistoryArray history_array;

    group::element_t canonical_sym = -1;
    search::node_pool_index_t node_index = -1;
    core::seat_index_t active_seat = -1;
    bool add_noise = false;
  };

  GeneralContext(const ManagerParams& mparams, search::mutex_vec_sptr_t node_mutex_pool);
  void clear();

  const ManagerParams manager_params;
  const SearchParams pondering_search_params;

  LookupTable lookup_table;
  RootInfo root_info;
  SearchParams search_params;
};

}  // namespace search

#include "inline/search/GeneralContext.inl"
