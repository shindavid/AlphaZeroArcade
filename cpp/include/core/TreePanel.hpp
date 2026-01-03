#pragma once

#include "core/BasicTypes.hpp"

namespace core {

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
  const tree_panel_vec_t& nodes() { return nodes_; }
  void add_node(game_tree_index_t new_node, game_tree_index_t parent_node,
                seat_index_t seat, action_mode_t action_mode);

 private:
  TreePanel();
  tree_panel_vec_t nodes_;
  int num_lanes_ = 1;
};

}  // namespace core
