#include "core/TreePanel.hpp"

#include "util/Asserts.hpp"

namespace core {

TreePanel* TreePanel::get_instance() {
  static TreePanel instance;
  return &instance;
}

TreePanel::TreePanel() {
  Node root{0, 0, -1, 0};
  nodes_.push_back(root);
}

const TreePanel::Node& TreePanel::node(game_tree_index_t ix) {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()),
                 "Invalid node index: ix={}, size={}", ix, nodes_.size());
  return nodes_[ix];
}

void TreePanel::add_node(game_tree_index_t new_node, game_tree_index_t parent_node,
                         seat_index_t seat, action_mode_t action_mode) {
  if (new_node < static_cast<game_tree_index_t>(nodes_.size())) {
    return;
  }

  RELEASE_ASSERT(new_node == static_cast<game_tree_index_t>(nodes_.size()),
                 "Nodes must be added in order: new_node={}, size={}", new_node, nodes_.size());

  int move = nodes_[parent_node].move + 1;

  /*
   * TODO: The current allocation is monotonic and will result in very sparse/wide trees.
   * Implement a layout algorithm (outside of add_node())that allows lane reuse for non-overlapping
   * branches.
   */
  int lane;
  if (nodes_[parent_node].first_child_ix < 0) {
    lane = nodes_[parent_node].lane;
  } else {
    lane = num_lanes_++;
  }

  nodes_.emplace_back(move, lane, seat, new_node, parent_node);

  if (nodes_[parent_node].first_child_ix < 0) {
    nodes_[parent_node].first_child_ix = new_node;
  } else {
    game_tree_index_t last_sibling = nodes_[parent_node].first_child_ix;
    while (nodes_[last_sibling].next_sibling_ix >= 0) {
      last_sibling = nodes_[last_sibling].next_sibling_ix;
    }
    nodes_[last_sibling].next_sibling_ix = new_node;
  }

}

}  // namespace core
