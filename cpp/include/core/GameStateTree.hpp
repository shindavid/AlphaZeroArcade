#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/CompactBitSet.hpp"

#include <vector>

namespace core {

template <concepts::Game Game>
class GameStateTree {
 public:
  using State = Game::State;
  using Rules = Game::Rules;
  using Constants = Game::Constants;

  struct AdvanceUpdate {
    game_tree_index_t ix = kNullNodeIx;
    action_t action = kNullAction;
    seat_index_t seat = -1;
    bool is_chance = false;
  };

  const State& state(game_tree_index_t ix) const;
  void init();
  game_tree_index_t advance(AdvanceUpdate update);
  game_tree_node_aux_t get_player_aux(game_tree_index_t ix, seat_index_t seat) const { return nodes_[ix].aux[seat]; }
  void set_player_aux(game_tree_index_t ix, seat_index_t seat, game_tree_node_aux_t aux) {
    nodes_[ix].aux[seat] = aux;
  }
  game_tree_index_t get_parent_index(game_tree_index_t ix) const;

 private:
  struct Node {
    const State state;
    const game_tree_index_t parent_ix;
    const action_t action_from_parent;
    game_tree_index_t first_child_ix = kNullNodeIx;
    game_tree_index_t next_sibling_ix = kNullNodeIx;
    seat_index_t seat = -1;
    bool is_chance = false;

    /*
     * Auxiliary data for players. Each player can store 8-byte data here for their private access.
     *
     * IMPORTANT NOTE: aux = 0 is reserved to mean "no aux data". Hence, players should avoid
     * storing aux = 0 here.
     */
    game_tree_node_aux_t aux[Constants::kNumPlayers] = {};

    util::CompactBitSet<Constants::kNumPlayers> player_acted;

    Node(const State& s, game_tree_index_t p = kNullNodeIx, action_t a = kNullAction)
        : state(s), parent_ix(p), action_from_parent(a) {}
  };
  std::vector<Node> nodes_;
};

}  // namespace core

#include "inline/core/GameStateTree.inl"
