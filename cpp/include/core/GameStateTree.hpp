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
  using PlayerActed = util::CompactBitSet<Constants::kNumPlayers>;

  const State& state(game_tree_index_t ix) const;
  void init();
  game_tree_index_t advance(game_tree_index_t from_ix, action_t action);
  game_tree_node_aux_t get_player_aux(game_tree_index_t ix, seat_index_t seat) const {
    return nodes_[ix].aux[seat];
  }
  void set_player_aux(game_tree_index_t ix, seat_index_t seat, game_tree_node_aux_t aux) {
    nodes_[ix].aux[seat] = aux;
  }
  game_tree_index_t get_parent_index(game_tree_index_t ix) const;
  seat_index_t get_parent_seat(game_tree_index_t ix) const;
  step_t get_step(game_tree_index_t ix) const { return nodes_[ix].step; }
  bool player_acted(game_tree_index_t ix, seat_index_t seat) const {
    return nodes_[ix].player_acted[seat];
  }
  seat_index_t get_active_seat(game_tree_index_t ix) const { return nodes_[ix].seat; }
  action_mode_t get_action_mode(game_tree_index_t ix) const { return nodes_[ix].action_mode; }
  action_t get_action(game_tree_index_t ix) const { return nodes_[ix].action_from_parent; }
  bool is_chance_node(game_tree_index_t ix) const;

 private:
  struct Node {
    const State state;
    const game_tree_index_t parent_ix = kNullNodeIx;
    const action_t action_from_parent = kNullAction;
    game_tree_index_t first_child_ix = kNullNodeIx;
    game_tree_index_t next_sibling_ix = kNullNodeIx;
    step_t step = -1;
    PlayerActed player_acted;
    seat_index_t seat = -1;
    action_mode_t action_mode = -1;

    /*
     * Auxiliary data for players. Each player can store 8-byte data here for their private access.
     *
     * IMPORTANT NOTE: aux = 0 is reserved to mean "no aux data". Hence, players should avoid
     * storing aux = 0 here.
     */
    game_tree_node_aux_t aux[Constants::kNumPlayers] = {};

    Node(const State& s, seat_index_t se, action_mode_t am)
        : state(s), step(0), seat(se), action_mode(am) {}

    Node(const State& s, game_tree_index_t p, action_t a, step_t st, seat_index_t se,
         action_mode_t am, PlayerActed pa)
        : state(s),
          parent_ix(p),
          action_from_parent(a),
          step(st),
          player_acted(pa),
          seat(se),
          action_mode(am) {}
  };

  std::vector<Node> nodes_;
};

}  // namespace core

#include "inline/core/GameStateTree.inl"
