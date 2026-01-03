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

  static TreePanel* get_instance() {
    static TreePanel instance;
    return &instance;
  }

  const tree_panel_vec_t& nodes() { return nodes_; }

  game_tree_index_t update(seat_index_t seat, game_tree_index_t new_node, action_mode_t action_mode);

 private:

  TreePanel() {
    Node root{0, 0, 0, 0};
    nodes_.push_back(root);
  }

  tree_panel_vec_t nodes_;
};

}  // namespace core
