#pragma once

#include "games/othello/Constants.hpp"

namespace othello {
/*
 * We split GameState into Core and Aux so that operator(<=>/==) and hashing can be defined entirely
 * in terms of Core. This way, GameState just dispatches its operators to the automatically
 * generated ones on Core.
 *
 * Core contains the bare minimum of information needed to apply the rules of the game.
 * Aux is intended to contain values derived from Core that accelerate rule calculations.
 *
 * In Othello, however, Aux currently holds data such as `stable_discs` that
 * do not assist with Rules. Instead, they leak details about input
 * tensorization into GameState.
 *
 * If InputTensorizor were reworked into a stateful object, rather than a  collection of static
 * methods,`stable_discs` could be moved out, and this Core/Aux split would no longer be
 * appropriate.
 */
struct GameState {
  auto operator<=>(const GameState& other) const { return core <=> other.core; }
  bool operator==(const GameState& other) const { return core == other.core; }
  size_t hash() const;
  int get_count(core::seat_index_t seat) const;
  core::seat_index_t get_player_at(int row, int col) const;  // -1 for unoccupied

  void compute_aux();
  void validate_aux() const {}

  struct Core {
    auto operator<=>(const Core& other) const = default;
    bool operator==(const Core& other) const = default;
    mask_t opponent_mask;    // spaces occupied by either player
    mask_t cur_player_mask;  // spaces occupied by current player
    core::seat_index_t cur_player;
    int8_t pass_count;
  };

  struct Aux {
    mask_t stable_discs;
  };
  Core core;
  Aux aux;
};

} // namespace othello

#include "inline/games/othello/GameState.inl"
