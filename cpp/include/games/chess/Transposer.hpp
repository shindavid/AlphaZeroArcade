#pragma once

#include "games/chess/GameState.hpp"

#include <cstdint>

namespace a0achess {

struct Transposer {
  using Key = uint64_t;
  static Key key(const GameState& state) { return state.hash(); }
};

}  // namespace a0achess
