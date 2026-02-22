#pragma once

#include "core/BasicTypes.hpp"
#include "games/chess/Constants.hpp"
#include "lc0/chess/board.h"
#include "util/StaticCircularBuffer.hpp"

namespace chess {

struct GameState {
  using ChessBoard = lczero::ChessBoard;
  using zobrist_hash_t = uint64_t;
  using history_hash_t = uint64_t;
  using ply_t = uint32_t;
  using CircularBuffer = util::StaticCircularBuffer<zobrist_hash_t, kNumRecentHashesToStore>;
  using seat_index_t = core::seat_index_t;

  auto operator<=>(const GameState& other) const { return history_hash <=> other.history_hash; };
  auto operator==(const GameState& other) const { return history_hash == other.history_hash; }
  size_t hash() const { return history_hash; }

  inline int count_repetitions() const;

  ChessBoard board;
  CircularBuffer recent_hashes;
  zobrist_hash_t zobrist_hash = 0;
  history_hash_t history_hash = 0;
  ply_t rule50_ply = 0;
};

}  // namespace chess

#include "inline/games/chess/GameState.inl"
