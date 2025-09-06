#include "games/connect4/InputTensorizor.hpp"

namespace c4 {

template <util::concepts::RandomAccessIteratorOf<c4::Game::State> Iter>
InputTensorizor::Tensor InputTensorizor::tensorize(Iter start, Iter cur) {
  core::seat_index_t cp = Game::Rules::get_current_player(*cur);
  Tensor tensor;
  tensor.setZero();
  int i = 0;
  Iter state = cur;
  while (true) {
    for (int row = 0; row < kNumRows; ++row) {
      for (int col = 0; col < kNumColumns; ++col) {
        core::seat_index_t p = state->get_player_at(row, col);
        if (p < 0) continue;
        int x = (kNumPlayers + cp - p) % kNumPlayers;
        tensor(i + x, row, col) = 1;
      }
    }
    if (state == start) break;
    state--;
    i += kNumPlayers;
  }
  return tensor;
}

}  // namespace c4
