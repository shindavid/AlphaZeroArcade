#pragma once

#include "core/InputTensorizor.hpp"
#include "core/Node.hpp"
#include "core/SimpleStateHistory.hpp"

#include <array>

namespace search {

template <typename Traits>
struct TraitsTypes {
  using NodeStableData = Traits::NodeStableData;
  using NodeStats = Traits::NodeStats;
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using State = Game::State;
  using EvalSpec = Traits::EvalSpec;

  using InputTensorizor = core::InputTensorizor<Game>;
  using StateHistory = core::SimpleStateHistory<State, InputTensorizor::kNumStatesToEncode - 1>;
  using StateHistoryArray = std::array<StateHistory, Game::SymmetryGroup::kOrder>;

  using Node = core::Node<NodeStableData, NodeStats>;

  struct Visitation {
    Node* node;
    Edge* edge;  // emanates from node, possibly nullptr
  };
};

}  // namespace search
