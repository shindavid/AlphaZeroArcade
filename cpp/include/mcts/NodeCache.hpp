#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/TypeDefs.hpp>
#include <util/CppUtil.hpp>

#include <map>
#include <mutex>
#include <unordered_map>

namespace mcts {

/*
 * Node lookup used to support MCGS.
 */
template <core::concepts::Game Game>
class NodeCache {
 public:
  using InputTensorizor = Game::InputTensorizor;
  using FullState = Game::FullState;
  using MCTSKey = InputTensorizor::MCTSKey;
  using ActionOutcome = Game::Types::ActionOutcome;
  using Node = mcts::Node<Game>;
  using Node_sptr = Node::sptr;

  void clear();
  void clear_before(move_number_t move_number);
  Node_sptr fetch_or_create(move_number_t move_number, const FullState& state,
                            const ActionOutcome& outcome, const ManagerParams* params);

 private:
  using submap_t = std::unordered_map<MCTSKey, Node_sptr>;
  using map_t = std::map<move_number_t, submap_t*>;

  map_t map_;
  std::mutex mutex_;
};

}  // namespace mcts

#include <inline/mcts/NodeCache.inl>
