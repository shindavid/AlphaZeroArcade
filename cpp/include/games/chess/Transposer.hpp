#pragma once

#include "games/chess/GameState.hpp"
#include "games/chess/TypeDefs.hpp"

namespace a0achess {

struct Transposer {
  using Key = history_hash_t;
  static Key key(const GameState& state) { return state.history_hash(); }
};

}  // namespace a0achess
