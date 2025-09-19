#pragma once

#include "core/Node.hpp"

#include <array>

namespace search {

template <typename Traits>
struct TraitsTypes {
  using NodeStableData = Traits::NodeStableData;
  using NodeStats = Traits::NodeStats;
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using EvalSpec = Traits::EvalSpec;

  using StateHistory = Game::StateHistory;
  using SymmetryGroup = Game::SymmetryGroup;
  using StateHistoryArray = std::array<StateHistory, SymmetryGroup::kOrder>;

  using Node = core::Node<NodeStableData, NodeStats>;

  struct Visitation {
    Node* node;
    Edge* edge;  // emanates from node, possibly nullptr
  };
};

}  // namespace search
