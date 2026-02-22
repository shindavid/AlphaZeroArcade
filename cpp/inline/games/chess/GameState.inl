#include "games/chess/GameState.hpp"

namespace chess {

inline int GameState::count_repetitions() const {
  if (recent_hashes.size() < 4) {
    return 0;
  }

  int repetitions = 0;
  for (int i = recent_hashes.size() - 2; i >= 0; i -= 2 ) {
    if (*(recent_hashes.begin() + i) == zobrist_hash) {
      repetitions++;
    }
  }
  return repetitions;
}

}  // namespace chess
