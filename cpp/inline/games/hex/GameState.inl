#include "games/hex/GameState.hpp"

#include "games/hex/Constants.hpp"
#include "games/hex/MaskReverser.hpp"
#include "util/CppUtil.hpp"
#include "util/Exceptions.hpp"

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

inline void GameState::Core::init() { std::memset(this, 0, sizeof(Core)); }

inline vertex_t GameState::Core::find_occupied(core::seat_index_t seat) const {
  for (int row = 0; row < Constants::kBoardDim; ++row) {
    mask_t mask = rows[seat][row];
    if (mask) {
      return row * Constants::kBoardDim + std::countr_zero(mask);
    }
  }
  throw util::Exception("Illegal call to {}()", __func__);
}

inline void GameState::Aux::init() {
  for (int i = 0; i < Constants::kNumPlayers; ++i) {
    union_find[i].init();
  }
}

inline core::seat_index_t GameState::get_player_at(int row, int col) const {
  int cp = core.cur_player;
  bool occupied_by_cur_player = (core.rows[cp][row] & (mask_t(1) << col)) != 0;
  bool occupied_by_other_player = (core.rows[1 - cp][row] & (mask_t(1) << col)) != 0;
  return occupied_by_other_player ? (1 - cp) : occupied_by_cur_player ? cp : -1;
}

}  // namespace hex
