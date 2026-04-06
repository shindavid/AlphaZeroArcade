#include "games/connect4/GameState.hpp"

namespace c4 {

inline void GameState::init() {
  full_mask = 0;
  cur_player_mask = 0;
}

}  // namespace c4
