#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/CompactBitSet.hpp"

#include <array>
#include <vector>

namespace core {

/*
 * Common tree-node bookkeeping fields shared by both perfect- and imperfect-info specializations.
 */
template <concepts::Game Game>
struct GameStateTreeNodeBase {
  using Move = Game::Move;
  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  using PlayerActed = util::CompactBitSet<kNumPlayers>;

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
  game_tree_node_aux_t aux[kNumPlayers] = {};

 protected:
  // Root-node base initializer
  GameStateTreeNodeBase(seat_index_t se) : step(0), seat(se) {}

  // Child-node base initializer
  GameStateTreeNodeBase(game_tree_index_t p, const Move& m, step_t st, seat_index_t se,
                        PlayerActed pa)
      : parent_ix(p),
        move_from_parent(m),
        step(st),
        player_acted(pa),
        move_from_parent_is_valid(true),
        seat(se) {}
};

/*
 * GameStateTreeNode is specialized by information_level_t.
 *
 * Perfect-info: stores only a State; info_set() returns it directly.
 * Imperfect-info: stores a State + per-seat InfoSet array.
 */
template <concepts::Game Game, information_level_t InfoLevel = Game::Types::kInformationLevel>
struct GameStateTreeNode;

template <concepts::Game Game>
struct GameStateTreeNode<Game, kPerfectInfo> : GameStateTreeNodeBase<Game> {
  using State = Game::State;
  using InfoSet = Game::InfoSet;
  using Move = Game::Move;
  using PlayerActed = GameStateTreeNodeBase<Game>::PlayerActed;

  const State state;

  const InfoSet& info_set(seat_index_t) const { return state; }

  // Root ctor
  GameStateTreeNode(const State& s, seat_index_t se) : GameStateTreeNodeBase<Game>(se), state(s) {}

  // Child ctor
  GameStateTreeNode(const State& s, game_tree_index_t p, const Move& m, step_t st, seat_index_t se,
                    PlayerActed pa)
      : GameStateTreeNodeBase<Game>(p, m, st, se, pa), state(s) {}
};

template <concepts::Game Game>
struct GameStateTreeNode<Game, kImperfectInfo> : GameStateTreeNodeBase<Game> {
  using State = Game::State;
  using InfoSet = Game::InfoSet;
  using Move = Game::Move;
  using PlayerActed = GameStateTreeNodeBase<Game>::PlayerActed;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  using InfoSetArray = std::array<InfoSet, kNumPlayers>;

  const State state;
  const InfoSetArray info_sets;

  const InfoSet& info_set(seat_index_t s) const { return info_sets[s]; }

  // Root ctor
  GameStateTreeNode(const State& s, InfoSetArray is, seat_index_t se)
      : GameStateTreeNodeBase<Game>(se), state(s), info_sets(std::move(is)) {}

  // Child ctor
  GameStateTreeNode(const State& s, InfoSetArray is, game_tree_index_t p, const Move& m, step_t st,
                    seat_index_t se, PlayerActed pa)
      : GameStateTreeNodeBase<Game>(p, m, st, se, pa), state(s), info_sets(std::move(is)) {}
};

/*
 * GameStateTree manages the game history as a tree of nodes. Templated on the Node type,
 * which is specialized by information_level_t.
 */
template <concepts::Game Game, information_level_t InfoLevel = Game::Types::kInformationLevel>
class GameStateTree {
 public:
  using Node = GameStateTreeNode<Game, InfoLevel>;
  using Move = Game::Move;
  using State = Game::State;
  using InfoSet = Game::InfoSet;
  using Rules = Game::Rules;
  using Constants = Game::Constants;

  static constexpr int kNumPlayers = Constants::kNumPlayers;

  const State& state(game_tree_index_t ix) const;
  const InfoSet& info_set(game_tree_index_t ix, seat_index_t seat) const;

  game_tree_node_aux_t get_player_aux(game_tree_index_t ix, seat_index_t seat) const {
    return nodes_[ix].aux[seat];
  }
  void set_player_aux(game_tree_index_t ix, seat_index_t seat, game_tree_node_aux_t aux) {
    nodes_[ix].aux[seat] = aux;
  }
  game_tree_index_t get_parent_index(game_tree_index_t ix) const;
  seat_index_t get_parent_seat(game_tree_index_t ix) const;
  step_t get_step(game_tree_index_t ix) const { return nodes_[ix].step; }
  const Move* get_move_from_parent(game_tree_index_t ix) const {
    return &nodes_[ix].move_from_parent;
  }
  bool player_acted(game_tree_index_t ix, seat_index_t seat) const {
    return nodes_[ix].player_acted[seat];
  }
  seat_index_t get_active_seat(game_tree_index_t ix) const { return nodes_[ix].seat; }
  const Move* get_move(game_tree_index_t ix) const {
    return nodes_[ix].move_from_parent_is_valid ? &nodes_[ix].move_from_parent : nullptr;
  }
  bool is_chance_node(game_tree_index_t ix) const;

  void init();
  game_tree_index_t advance(game_tree_index_t from_ix, const Move& move);

  std::vector<Node> nodes_;

 private:
  game_tree_index_t find_child(game_tree_index_t from_ix, const Move& move,
                               game_tree_index_t& last_child_ix) const;
  void link_child(game_tree_index_t from_ix, game_tree_index_t new_ix,
                  game_tree_index_t last_child_ix);
};

}  // namespace core

#include "inline/core/GameStateTree.inl"
