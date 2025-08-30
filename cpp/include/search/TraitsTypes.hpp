#pragma once

#include "search/LookupTable.hpp"

#include <array>
#include <vector>

namespace search {

template <typename Traits>
struct TraitsTypes {
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using ManagerParams = Traits::ManagerParams;
  using AuxState = Traits::AuxState;

  using StateHistory = Game::StateHistory;
  using SymmetryGroup = Game::SymmetryGroup;
  using StateHistoryArray = std::array<StateHistory, SymmetryGroup::kOrder>;

  struct Visitation {
    Node* node;
    Edge* edge;  // emanates from node, possibly nullptr
  };
  using search_path_t = std::vector<Visitation>;

  using LookupTable = search::LookupTable<Traits>;
};

}  // namespace search
