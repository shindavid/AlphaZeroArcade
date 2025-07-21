#pragma once

#include "core/BasicTypes.hpp"
#include "games/hex/Constants.hpp"
#include "games/hex/UnionFind.hpp"
#include "util/CppUtil.hpp"

namespace hex {

/*
 * GameState is split internally into two parts: Core and Aux.
 *
 * Core unamgibuously represents the game state.
 *
 * Aux contains additional information that can be computed from Core, but is stored for
 * efficiency.
 *
 * TODO: use Zobrist-hashing to speed-up hashing.
 */
struct GameState {
  auto operator<=>(const GameState& other) const { return core <=> other.core; }
  bool operator==(const GameState& other) const { return core == other.core; }
  bool operator!=(const GameState& other) const { return core != other.core; }
  size_t hash() const { return util::PODHash<Core>{}(core); }
  void init();
  void rotate();

  // Core unambiguously represents the game state.
  struct Core {
    auto operator<=>(const Core& other) const = default;
    void init();

    // Assumes that at least one vertex is occupied by the given player, and returns any such vertex
    vertex_t find_occupied(core::seat_index_t seat) const;

    mask_t rows[Constants::kNumPlayers][Constants::kBoardDim];

    // Without the swap-rule, cur_player can be derived from the board. But with the swap-rule,
    // we need to store it explicitly.
    core::seat_index_t cur_player;
    bool post_swap_phase;
  };

  struct Aux {
    auto operator<=>(const Aux&) const = default;
    void init();

    UnionFind union_find[Constants::kNumPlayers];
  };

  Core core;
  Aux aux;
};

}  // namespace hex

namespace std {

template <>
struct hash<hex::GameState> {
  size_t operator()(const hex::GameState& pos) const { return pos.hash(); }
};

}  // namespace std

#include "inline/games/hex/GameState.inl"
