#pragma once

#include "core/BasicTypes.hpp"
#include "core/StateIterator.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

namespace search {

// GeneralContext<SearchSpec> contains data members that apply to the entire game tree.
template <search::concepts::SearchSpec SearchSpec>
struct GeneralContext {
  using Node = SearchSpec::Node;

  using Game = SearchSpec::Game;
  using ManagerParams = SearchSpec::ManagerParams;
  using AuxState = SearchSpec::AuxState;

  using Rules = Game::Rules;
  using State = Game::State;

  using LookupTable = search::LookupTable<SearchSpec>;
  using EvalSpec = SearchSpec::EvalSpec;
  using InputEncoder = EvalSpec::TensorEncodings::InputEncoder;
  using StateIterator = core::StateIterator<Game>;

  struct RootInfo {
    void clear();

    State state;
    InputEncoder input_encoder;
    int state_step = 0;  // incremented every time state changes
    core::node_pool_index_t node_index = -1;
    core::seat_index_t active_seat = -1;
    bool add_noise = false;
  };

  GeneralContext(const ManagerParams& mparams, core::mutex_vec_sptr_t node_mutex_pool);
  void clear();
  void step();
  void jump_to(StateIterator it, core::step_t step);
  Node* root() const { return lookup_table.get_node(root_info.node_index); }

  const ManagerParams manager_params;
  const SearchParams pondering_search_params;

  AuxState aux_state;
  LookupTable lookup_table;
  RootInfo root_info;
  SearchParams search_params;
};

}  // namespace search

#include "inline/search/GeneralContext.inl"
