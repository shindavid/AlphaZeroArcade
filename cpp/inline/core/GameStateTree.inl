#include "core/GameStateTree.hpp"

#include "core/BasicTypes.hpp"

namespace core {

template <concepts::Game Game, information_level_t InfoLevel>
const typename GameStateTree<Game, InfoLevel>::State&
GameStateTree<Game, InfoLevel>::state(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return nodes_[ix].state;
}

template <concepts::Game Game, information_level_t InfoLevel>
const typename GameStateTree<Game, InfoLevel>::InfoSet&
GameStateTree<Game, InfoLevel>::info_set(game_tree_index_t ix, seat_index_t seat) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return nodes_[ix].info_set(seat);
}

template <concepts::Game Game, information_level_t InfoLevel>
bool GameStateTree<Game, InfoLevel>::is_chance_node(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return Rules::is_chance_state(nodes_[ix].state);
}

template <concepts::Game Game, information_level_t InfoLevel>
game_tree_index_t GameStateTree<Game, InfoLevel>::get_parent_index(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  return nodes_[ix].parent_ix;
}

template <concepts::Game Game, information_level_t InfoLevel>
seat_index_t GameStateTree<Game, InfoLevel>::get_parent_seat(game_tree_index_t ix) const {
  RELEASE_ASSERT(ix >= 0 && ix < static_cast<game_tree_index_t>(nodes_.size()));
  game_tree_index_t parent_ix = nodes_[ix].parent_ix;
  if (parent_ix < 0) {
    return -1;
  } else {
    return nodes_[parent_ix].seat;
  }
}

template <concepts::Game Game, information_level_t InfoLevel>
game_tree_index_t GameStateTree<Game, InfoLevel>::find_child(
  game_tree_index_t from_ix, const Move& move, game_tree_index_t& last_child_ix) const {
  RELEASE_ASSERT(from_ix >= 0 && from_ix < static_cast<game_tree_index_t>(nodes_.size()));
  last_child_ix = kNullNodeIx;
  for (game_tree_index_t i = nodes_[from_ix].first_child_ix; i != kNullNodeIx;
       i = nodes_[i].next_sibling_ix) {
    if (move == nodes_[i].move_from_parent) {
      return i;
    }
    last_child_ix = i;
  }
  return kNullNodeIx;
}

template <concepts::Game Game, information_level_t InfoLevel>
void GameStateTree<Game, InfoLevel>::link_child(
  game_tree_index_t from_ix, game_tree_index_t new_ix, game_tree_index_t last_child_ix) {
  if (nodes_[from_ix].first_child_ix == kNullNodeIx) {
    nodes_[from_ix].first_child_ix = new_ix;
  } else {
    RELEASE_ASSERT(last_child_ix != kNullNodeIx);
    nodes_[last_child_ix].next_sibling_ix = new_ix;
  }
}

template <concepts::Game Game, information_level_t InfoLevel>
void GameStateTree<Game, InfoLevel>::init() {
  nodes_.clear();
  State state;
  Rules::init_state(state);
  seat_index_t seat = Rules::get_current_player(state);

  if constexpr (InfoLevel == kImperfectInfo) {
    using InfoSetArray = Node::InfoSetArray;
    InfoSetArray info_sets;
    for (int s = 0; s < kNumPlayers; ++s) {
      info_sets[s] = Rules::state_to_info_set(state, s);
    }
    nodes_.emplace_back(state, std::move(info_sets), seat);
  } else {
    nodes_.emplace_back(state, seat);
  }
}

template <concepts::Game Game, information_level_t InfoLevel>
game_tree_index_t GameStateTree<Game, InfoLevel>::advance(
  game_tree_index_t from_ix, const Move& move) {
  game_tree_index_t last_child_ix;
  game_tree_index_t existing = find_child(from_ix, move, last_child_ix);
  if (existing != kNullNodeIx) return existing;

  game_tree_index_t new_ix = nodes_.size();
  link_child(from_ix, new_ix, last_child_ix);

  const State& parent_state = nodes_[from_ix].state;
  State new_state = parent_state;
  Rules::apply(new_state, move);

  seat_index_t seat = Rules::get_current_player(new_state);

  auto player_acted = nodes_[from_ix].player_acted;
  if (!Rules::is_chance_state(parent_state)) {
    seat_index_t parent_seat = nodes_[from_ix].seat;
    player_acted.set(parent_seat);
  }

  step_t step = nodes_[from_ix].step + 1;

  if constexpr (InfoLevel == kImperfectInfo) {
    using InfoSetArray = Node::InfoSetArray;
    InfoSetArray info_sets;
    for (int s = 0; s < kNumPlayers; ++s) {
      info_sets[s] = Rules::state_to_info_set(new_state, s);
    }
    nodes_.emplace_back(new_state, std::move(info_sets), from_ix, move, step, seat, player_acted);
  } else {
    nodes_.emplace_back(new_state, from_ix, move, step, seat, player_acted);
  }
  return new_ix;
}

}  // namespace core
