#pragma once

#include "core/BasicTypes.hpp"

#include <vector>

namespace core {

/*
 * TreePanel maintains a simplified shadow-copy of the game tree for UI visualization.
 * It maps game_tree_index_t to a layout using "moves" (depth) and "lanes" (width). This is
 * implemented as a singleton assuming that only one game is active at a time. If multiple games
 * are active simultanenously, we will need to extend this to support multiple TreePanel instances.
 */
class TreePanel {
 public:
  struct Node {
    int move;
    int lane;
    seat_index_t seat;
    game_tree_index_t index;
    game_tree_index_t parent_ix = kNullNodeIx;
    game_tree_index_t first_child_ix = kNullNodeIx;
    game_tree_index_t next_sibling_ix = kNullNodeIx;
  };

  // look up by game_tree_index_t
  using tree_panel_vec_t = std::vector<Node>;

  static TreePanel* get_instance();
  const Node& node(game_tree_index_t ix);
  void add_node(game_tree_index_t new_node, game_tree_index_t parent_node,
                seat_index_t seat, action_mode_t action_mode);

 private:
  TreePanel();
  tree_panel_vec_t nodes_;

  // Number of lanes allocated for the tree layout. Used to assign vertical offsets in add_node().
  int num_lanes_ = 1;
};

}  // namespace core
