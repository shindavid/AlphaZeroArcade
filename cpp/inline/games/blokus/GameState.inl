#include "games/blokus/GameState.hpp"
#include "util/CppUtil.hpp"

namespace blokus {

inline size_t GameState::hash() const { return util::PODHash<Core>{}(core); }

inline int GameState::remaining_square_count(color_t c) const {
  return kNumSquaresPerColor - core.occupied_locations[c].count();
}

}  // namespace blokus
