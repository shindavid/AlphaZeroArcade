#pragma once

#include "games/connect4/InputFrame.hpp"
#include "games/connect4/Move.hpp"

#include <functional>

namespace c4 {

struct GameState : public InputFrame {
  void init();
  Move last_move;
};

}  // namespace c4

namespace std {

template <>
struct hash<c4::GameState> {
  size_t operator()(const c4::GameState& state) const { return state.hash(); }
};

}  // namespace std

#include "inline/games/connect4/GameState.inl"
