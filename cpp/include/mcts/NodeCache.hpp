#pragma once

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Node.hpp>
#include <mcts/TypeDefs.hpp>

#include <map>
#include <mutex>
#include <unordered_map>

namespace mcts {

/*
 * Node lookup used to support MCGS.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class NodeCache {
 public:
  using GameStateTypes = core::GameStateTypes<GameState>;
  using Action = typename GameStateTypes::Action;
  using Node = mcts::Node<GameState, Tensorizor>;
  using Node_sptr = typename Node::sptr;

  void clear();
  void clear_before(move_number_t move_number);
  Node_sptr fetch_or_create(move_number_t move_number, Node* parent, const Action& action);

 private:
  using submap_t = std::unordered_map<GameState, Node_sptr>;
  using map_t = std::map<move_number_t, submap_t*>;

  map_t map_;
  std::mutex mutex_;
};

}  // namespace mcts

#include <mcts/inl/NodeCache.inl>
