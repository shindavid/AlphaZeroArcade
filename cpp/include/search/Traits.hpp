#pragma once

#include "search/LookupTable.hpp"
#include "search/concepts/AuxStateConcept.hpp"
#include "search/concepts/EdgeConcept.hpp"
#include "search/concepts/GraphTraitsConcept.hpp"
#include "search/concepts/ManagerParamsConcept.hpp"
#include "search/concepts/NodeConcept.hpp"

#include <array>
#include <vector>

namespace search {

template <core::concepts::Game G, search::concepts::Node<G> N, search::concepts::Edge E>
struct GraphTraits {
  using Game = G;
  using Node = N;
  using Edge = E;

  // static_assert(search::concepts::GraphTraits<GraphTraits>);
};

template <search::concepts::GraphTraits GT, search::concepts::ManagerParams MP,
          search::concepts::AuxState<MP> AS>
struct GeneralContextTraits {
  using Game = GT::Game;
  using Node = GT::Node;
  using Edge = GT::Edge;
  using GraphTraits = GT;
  using ManagerParams = MP;
  using AuxState = AS;

  // static_assert(search::concepts::GeneralContextTraits<GeneralContextTraits>);
};

template <typename Traits>
struct TraitsTypes {
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using ManagerParams = Traits::ManagerParams;
  using AuxState = Traits::AuxState;

  using GraphTraits = search::GraphTraits<Game, Node, Edge>;
  using GeneralContextTraits = search::GeneralContextTraits<GraphTraits, ManagerParams, AuxState>;

  using StateHistory = Game::StateHistory;
  using SymmetryGroup = Game::SymmetryGroup;
  using StateHistoryArray = std::array<StateHistory, SymmetryGroup::kOrder>;

  struct Visitation {
    Node* node;
    Edge* edge;  // emanates from node, possibly nullptr
  };
  using search_path_t = std::vector<Visitation>;

  using LookupTable = search::LookupTable<GraphTraits>;
};

}  // namespace search
