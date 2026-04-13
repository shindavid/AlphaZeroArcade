#pragma once

#include "alpha0/AuxState.hpp"
#include "alpha0/ManagerParams.hpp"
#include "alpha0/NodeStableData.hpp"
#include "alpha0/NodeStats.hpp"
#include "alpha0/Spec.hpp"
#include "core/BasicTypes.hpp"
#include "core/Node.hpp"
#include "core/StateIterator.hpp"
#include "alpha0/concepts/EvalSpecConcept.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"

namespace alpha0 {

// GeneralContext<EvalSpec> contains data members that apply to the entire game tree.
template <alpha0::concepts::EvalSpec EvalSpec>
struct GeneralContext {
  using Spec = alpha0::Spec<typename EvalSpec::Game, EvalSpec>;
  using Node = Spec::Node;

  using Game = EvalSpec::Game;
  using ManagerParams = alpha0::ManagerParams<EvalSpec>;
  using AuxState = alpha0::AuxState<ManagerParams>;

  using Rules = Game::Rules;
  using State = Game::State;

  using LookupTable = search::LookupTable<Spec>;
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
  const search::SearchParams pondering_search_params;

  AuxState aux_state;
  LookupTable lookup_table;
  RootInfo root_info;
  search::SearchParams search_params;
};

}  // namespace alpha0

#include "inline/alpha0/GeneralContext.inl"
