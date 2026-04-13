#pragma once

#include "alpha0/AuxState.hpp"
#include "alpha0/ManagerParams.hpp"
#include "alpha0/NodeStableData.hpp"
#include "alpha0/NodeStats.hpp"
#include "alpha0/Node.hpp"
#include "core/BasicTypes.hpp"
#include "core/StateIterator.hpp"
#include "alpha0/concepts/SpecConcept.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"

namespace alpha0 {

// GeneralContext<Spec> contains data members that apply to the entire game tree.
template <alpha0::concepts::Spec Spec>
struct GeneralContext {
  using Node = alpha0::Node<Spec>;

  using Game = Spec::Game;
  using ManagerParams = alpha0::ManagerParams<Spec>;
  using AuxState = alpha0::AuxState<ManagerParams>;

  using Rules = Game::Rules;
  using State = Game::State;

  using LookupTable = search::LookupTable<Spec>;
  using InputEncoder = Spec::TensorEncodings::InputEncoder;
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
  const search::SearchParams pondering_search_params;

  AuxState aux_state;
  LookupTable lookup_table;
  RootInfo root_info;
  search::SearchParams search_params;
};

}  // namespace alpha0

#include "inline/alpha0/GeneralContext.inl"
