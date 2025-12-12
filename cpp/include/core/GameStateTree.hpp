#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

#include <cstdint>
#include <vector>

namespace core {

template <concepts::Game Game>
class GameStateTree {
 public:
  using node_ix_t = int32_t;
  using State = Game::State;
  using Rules = Game::Rules;
  using Constants = Game::Constants;

  static constexpr node_ix_t kNullNodeIx = -1;
  static constexpr action_t kNullAction = -1;

  const State& state(node_ix_t ix) const;
  void init();
  node_ix_t advance(node_ix_t ix, action_t action);

 private:
  struct Node {
    const State state;
    const node_ix_t parent_ix;
    const action_t action_from_parent;
    node_ix_t first_child_ix = kNullNodeIx;
    node_ix_t next_sibling_ix = kNullNodeIx;

    // Auxilary data for players. Each player can store a pointer here for their *private* reference.
    void* aux[Constants::kNumPlayers];

    Node(const State& s, node_ix_t p = kNullNodeIx, action_t a = kNullAction)
        : state(s), parent_ix(p), action_from_parent(a) {}
  };
  std::vector<Node> nodes_;
};

}  // namespace core

#include "inline/core/GameStateTree.inl"
