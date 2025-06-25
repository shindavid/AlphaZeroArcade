#include <games/hex/GameState.hpp>

#include <cstring>

namespace hex {

inline void GameState::init() {
  core.init();
  aux.init();
}

inline void GameState::Core::init() {
  std::memset(this, 0, sizeof(Core));
}

inline void GameState::Aux::init() {
  for (int i = 0; i < Constants::kNumPlayers; ++i) {
    union_find[i].init();
  }
}

}  // namespace hex
