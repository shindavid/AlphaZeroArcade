#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/CompactBitSet.hpp"

#include <vector>

namespace core {

template <concepts::Game Game>
class GameStateTree {
 public:
  using Move = Game::Move;
  using State = Game::State;
  using Rules = Game::Rules;
  using Constants = Game::Constants;
  using PlayerActed = util::CompactBitSet<Constants::kNumPlayers>;

  const State& state(game_tree_index_t ix) const;
  void init();
  game_tree_index_t advance(game_tree_index_t from_ix, const Move& move);
  game_tree_node_aux_t get_player_aux(game_tree_index_t ix, seat_index_t seat) const {
    return nodes_[ix].aux[seat];
  }
  void set_player_aux(game_tree_index_t ix, seat_index_t seat, game_tree_node_aux_t aux) {
    nodes_[ix].aux[seat] = aux;
  }
  game_tree_index_t get_parent_index(game_tree_index_t ix) const;
  seat_index_t get_parent_seat(game_tree_index_t ix) const;
  step_t get_step(game_tree_index_t ix) const { return nodes_[ix].step; }
  const Move& get_move_from_parent(game_tree_index_t ix) const { return nodes_[ix].move_from_parent; }
  bool player_acted(game_tree_index_t ix, seat_index_t seat) const {
    return nodes_[ix].player_acted[seat];
  }
  seat_index_t get_active_seat(game_tree_index_t ix) const { return nodes_[ix].seat; }
  const Move* get_move(game_tree_index_t ix) const {
    return nodes_[ix].move_from_parent_is_valid ? &nodes_[ix].move_from_parent : nullptr;
  }
  bool is_chance_node(game_tree_index_t ix) const;

 private:
  struct Node {
    const State state;
    const game_tree_index_t parent_ix = kNullNodeIx;
    Move move_from_parent;
    game_tree_index_t first_child_ix = kNullNodeIx;
    game_tree_index_t next_sibling_ix = kNullNodeIx;
    step_t step = -1;
    PlayerActed player_acted;
    bool move_from_parent_is_valid = false;
    seat_index_t seat = -1;

    /*
     * Auxiliary data for players. Each player can store 8-byte data here for their private access.
     *
     * IMPORTANT NOTE: aux = 0 is reserved to mean "no aux data". Hence, players should avoid
     * storing aux = 0 here.
     */
    game_tree_node_aux_t aux[Constants::kNumPlayers] = {};

    // For starting position of game
    Node(const State& s, seat_index_t se) : state(s), step(0), seat(se) {}

    Node(const State& s, game_tree_index_t p, const Move& m, step_t st, seat_index_t se,
         PlayerActed pa)
        : state(s),
          parent_ix(p),
          move_from_parent(m),
          step(st),
          player_acted(pa),
          move_from_parent_is_valid(true),
          seat(se) {}
  };

  std::vector<Node> nodes_;
};

}  // namespace core

#include "inline/core/GameStateTree.inl"
