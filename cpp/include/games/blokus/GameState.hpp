#pragma once

#include <games/blokus/Constants.hpp>
#include <games/blokus/Types.hpp>

#include <cstdint>

namespace blokus {

/*
 * GameState is split internally into two parts: core_t and aux_t.
 *
 * core_t unamgibuously represents the game state.
 *
 * aux_t contains additional information that can be computed from core_t, but is stored for
 * efficiency.
 *
 * TODO: use Zobrist-hashing to speed-up hashing.
 */
struct GameState {
  auto operator<=>(const GameState& other) const { return core <=> other.core; }
  bool operator==(const GameState& other) const { return core == other.core; }
  bool operator!=(const GameState& other) const { return core != other.core; }
  size_t hash() const;
  int remaining_square_count(color_t) const;
  color_t last_placed_piece_color() const;
  int pass_count() const { return core.pass_count; }

  /*
   * Sets this->aux from this->core.
   */
  void compute_aux();

  /*
   * Throws an exception if aux is not consistent with core.
   */
  void validate_aux() const;

  // core_t unambiguously represents the game state.
  struct core_t {
    auto operator<=>(const core_t& other) const = default;
    BitBoard occupied_locations[kNumColors];

    color_t cur_color;
    int8_t pass_count;

    // We split the move into multiple parts:
    // 1. The location of a piece corner
    // 2. The piece/orientation/square to place on that location
    //
    // This value stores part 1. A location of (-1, -1) indicates that the player has not
    // selected a location.
    Location partial_move;
  };

  // TODO: consider moving some of these members into StateHistory. The ones that support
  // tensorization should stay here, but the ones that only facilitate rules-calculations can
  // be moved to StateHistory to reduce the disk footprint of game logs.
  struct aux_t {
    auto operator<=>(const aux_t&) const = default;
    PieceMask played_pieces[kNumColors];
    BitBoard unplayable_locations[kNumColors];
    BitBoard corner_locations[kNumColors];
  };

  core_t core;
  aux_t aux;
};

}  // namespace blokus

#include <inline/games/blokus/GameState.inl>