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
const GameStateTree<Game>::State& GameStateTree<Game>::state(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return nodes_[ix].state;
}

template <concepts::Game Game>
game_tree_index_t GameStateTree<Game>::advance(const StateChangeUpdate& update) {
  auto ix = update.game_tree_index;
  auto action = update.action;
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  RELEASE_ASSERT(update.seat >= 0 && update.seat < Constants::kNumPlayers);

  game_tree_index_t last_child_ix = kNullNodeIx;
  for (game_tree_index_t i = nodes_[ix].first_child_ix; i != kNullNodeIx;
       i = nodes_[i].next_sibling_ix) {
    if (action == nodes_[i].action_from_parent) {
      return i;
    }
    last_child_ix = i;
  }

  game_tree_index_t new_ix = nodes_.size();
  if (nodes_[ix].first_child_ix == kNullNodeIx) {
    nodes_[ix].first_child_ix = new_ix;
    nodes_[ix].seat = update.seat;
    nodes_[ix].is_chance = Game::Rules::is_chance_mode(update.action_mode);
  }
  RELEASE_ASSERT(nodes_[ix].seat == update.seat);
  RELEASE_ASSERT(nodes_[ix].is_chance == Game::Rules::is_chance_mode(update.action_mode));

  if (last_child_ix != kNullNodeIx) {
    nodes_[last_child_ix].next_sibling_ix = new_ix;
  }

  State new_state = nodes_[ix].state;
  Rules::apply(new_state, action);

  auto player_acted = nodes_[ix].player_acted;
  if (!Game::Rules::is_chance_mode(update.action_mode)) {
    player_acted.set(update.seat);
  }
  nodes_.emplace_back(new_state, ix, action, player_acted);
  return new_ix;
}

template <concepts::Game Game>
game_tree_index_t GameStateTree<Game>::get_parent_index(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return nodes_[ix].parent_ix;
}

}  // namespace core
