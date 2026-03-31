#include "core/GameStateTree.hpp"

#include "core/BasicTypes.hpp"

namespace core {

template <concepts::Game Game>
void GameStateTree<Game>::init() {
  nodes_.clear();
  State state;
  Rules::init_state(state);
  seat_index_t seat = Rules::get_current_player(state);
  game_phase_t game_phase = Rules::get_game_phase(state);
  nodes_.emplace_back(state, seat, game_phase);
}

template <concepts::Game Game>
const GameStateTree<Game>::State& GameStateTree<Game>::state(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return nodes_[ix].state;
}

template <concepts::Game Game>
bool GameStateTree<Game>::is_chance_node(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return Rules::is_chance_phase(nodes_[ix].game_phase);
}

template <concepts::Game Game>
game_tree_index_t GameStateTree<Game>::advance(game_tree_index_t from_ix, const Move& move) {
  RELEASE_ASSERT(from_ix >= 0 && from_ix < static_cast<game_tree_index_t>(nodes_.size()));

  game_tree_index_t last_child_ix = kNullNodeIx;
  for (game_tree_index_t i = nodes_[from_ix].first_child_ix; i != kNullNodeIx;
       i = nodes_[i].next_sibling_ix) {
    if (move == nodes_[i].move_from_parent) {
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
  Rules::apply(new_state, move);

  seat_index_t seat = Rules::get_current_player(new_state);
  game_phase_t game_phase = Rules::get_game_phase(new_state);

  auto player_acted = nodes_[from_ix].player_acted;
  seat_index_t parent_seat = nodes_[from_ix].seat;
  game_phase_t parent_game_phase = nodes_[from_ix].game_phase;
  if (!Rules::is_chance_phase(parent_game_phase)) {
    player_acted.set(parent_seat);
  }

  step_t step = nodes_[from_ix].step + 1;

  nodes_.emplace_back(new_state, from_ix, move, step, seat, game_phase, player_acted);
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
