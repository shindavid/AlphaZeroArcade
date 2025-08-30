#pragma once

#include "search/LookupTable.hpp"

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

template <typename Traits>
struct TraitsTypes {
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using GraphTraits = search::GraphTraits<Game, Node, Edge>;

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
