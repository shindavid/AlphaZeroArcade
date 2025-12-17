#include "core/GameStateTree.hpp"

namespace core {

template <concepts::Game Game>
void GameStateTree<Game>::init() {
  nodes_.clear();
  State state;
  Rules::init_state(state);
  nodes_.emplace_back(state);
}

template <concepts::Game Game>
const GameStateTree<Game>::State& GameStateTree<Game>::state(node_ix_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<node_ix_t>(nodes_.size()));
  return nodes_[ix].state;
}

template <concepts::Game Game>
node_ix_t GameStateTree<Game>::advance(node_ix_t ix, action_t action) {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<node_ix_t>(nodes_.size()));
  node_ix_t last_child_ix = kNullNodeIx;
  for (node_ix_t i = nodes_[ix].first_child_ix; i != kNullNodeIx; i = nodes_[i].next_sibling_ix) {
    if (action == nodes_[i].action_from_parent) {
      return i;
    }

    if (nodes_[i].next_sibling_ix == kNullNodeIx) {
      last_child_ix = i;
    }
  }

  State new_state = nodes_[ix].state;
  Rules::apply(new_state, action);

  nodes_.emplace_back(new_state, ix, action);
  node_ix_t new_ix = nodes_.size() - 1;

  if (nodes_[ix].first_child_ix == kNullNodeIx) {
    nodes_[ix].first_child_ix = new_ix;
  }

  if (last_child_ix != kNullNodeIx) {
    nodes_[last_child_ix].next_sibling_ix = new_ix;
  }

  return new_ix;
}

}  // namespace core
