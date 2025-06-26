#include <games/hex/GameState.hpp>

#include <games/hex/Constants.hpp>
#include <games/hex/MaskReverser.hpp>
#include <util/CppUtil.hpp>

#include <cstring>

namespace hex {

inline void GameState::init() {
  core.init();
  aux.init();
}

inline void GameState::rotate() {
  constexpr int B = Constants::kBoardDim;
  constexpr int P = Constants::kNumPlayers;

  for (int p = 0; p < P; ++p) {
    mask_t tmp[B];
    for (int r = 0; r < B; ++r) {
      tmp[r] = MaskReverser::reverse(core.rows[p][B - 1 - r]);
    }
    std::copy(tmp, tmp + B, core.rows[p]);
    aux.union_find[p].rotate();
  }
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
