#pragma once

#include <array>
#include <vector>

namespace search {

template <typename Traits>
struct TraitsTypes {
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using Game = Traits::Game;

  using StateHistory = Game::StateHistory;
  using SymmetryGroup = Game::SymmetryGroup;
  using StateHistoryArray = std::array<StateHistory, SymmetryGroup::kOrder>;

  struct Visitation {
    Node* node;
    Edge* edge;  // emanates from node, possibly nullptr
  };
  using search_path_t = std::vector<Visitation>;
};

}  // namespace search
