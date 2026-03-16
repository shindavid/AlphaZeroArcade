#pragma once

#include "games/hex/GameState.hpp"

namespace hex {

struct Transposer {
  using Key = GameState::Core;
  static Key key(const GameState& state) { return state.core; }
};

}  // namespace hex
