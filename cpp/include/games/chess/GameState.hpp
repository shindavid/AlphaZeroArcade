#pragma once

#include "chess-library/include/chess.hpp"
#include "core/BasicTypes.hpp"

#include <cstdint>
#include <functional>

namespace a0achess {

class GameState : public chess::Board {
 public:
  using chess::Board::Board;
  using ProtectedCtor = chess::Board::ProtectedCtor;
  using zobrist_hash_t = uint64_t;

  void backtrack_to(const GameState& prev_state);
  core::action_t action_from_uci(const std::string& uci) const;

  friend struct InputFrame;
};

}  // namespace a0achess

namespace std {

template <>
struct hash<a0achess::GameState> {
  size_t operator()(const a0achess::GameState& state) const { return state.hash(); }
};

}  // namespace std

#include "inline/games/chess/GameState.inl"
