#include "core/GameStateTree.hpp"

#include "core/BasicTypes.hpp"

namespace core {

template <concepts::Game Game>
void GameStateTree<Game>::init() {
  nodes_.clear();
  State state;
  Rules::init_state(state);
  seat_index_t seat = Rules::get_current_player(state);
  action_mode_t action_mode = Rules::get_action_mode(state);
  nodes_.emplace_back(state, seat, action_mode);
}

template <concepts::Game Game>
const GameStateTree<Game>::State& GameStateTree<Game>::state(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return nodes_[ix].state;
}

template <concepts::Game Game>
bool GameStateTree<Game>::is_chance_node(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return Rules::is_chance_mode(nodes_[ix].action_mode);
}

template <concepts::Game Game>
game_tree_index_t GameStateTree<Game>::advance(game_tree_index_t from_ix, action_t action) {
  RELEASE_ASSERT(from_ix >= 0 && from_ix < static_cast<game_tree_index_t>(nodes_.size()));

  game_tree_index_t last_child_ix = kNullNodeIx;
  for (game_tree_index_t i = nodes_[from_ix].first_child_ix; i != kNullNodeIx;
       i = nodes_[i].next_sibling_ix) {
    if (action == nodes_[i].action_from_parent) {
      return i;
    }
    last_child_ix = i;
  }

  game_tree_index_t new_ix = nodes_.size();
  if (nodes_[from_ix].first_child_ix == kNullNodeIx) {
    nodes_[from_ix].first_child_ix = new_ix;
  } else {
    RELEASE_ASSERT(last_child_ix != kNullNodeIx);
    nodes_[last_child_ix].next_sibling_ix = new_ix;
  }

  State new_state = nodes_[from_ix].state;
  Rules::apply(new_state, action);

  seat_index_t seat = Rules::get_current_player(new_state);
  action_mode_t action_mode = Rules::get_action_mode(new_state);

  auto player_acted = nodes_[from_ix].player_acted;
  seat_index_t parent_seat = nodes_[from_ix].seat;
  bool parent_is_chance = Rules::is_chance_mode(nodes_[from_ix].action_mode);
  if (!parent_is_chance) {
    player_acted.set(parent_seat);
  }

  step_t step = nodes_[from_ix].step + 1;

  nodes_.emplace_back(new_state, from_ix, action, step, seat, action_mode, player_acted);
  return new_ix;
}

template <concepts::Game Game>
game_tree_index_t GameStateTree<Game>::get_parent_index(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return nodes_[ix].parent_ix;
}

template <concepts::Game Game>
seat_index_t GameStateTree<Game>::get_parent_seat(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  game_tree_index_t parent_ix = nodes_[ix].parent_ix;
  if (parent_ix < 0) {
    return -1;
  } else {
    return nodes_[parent_ix].seat;
  }
}

}  // namespace core
