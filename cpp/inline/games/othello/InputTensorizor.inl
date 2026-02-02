#include "games/othello/InputTensorizor.hpp"

namespace othello {

template <util::concepts::RandomAccessIteratorOf<Game::State> Iter>
InputTensorizor::Tensor InputTensorizor::tensorize(Iter start, Iter cur) {
  core::seat_index_t cp = Game::Rules::get_current_player(*cur);
  Tensor tensor;
  tensor.setZero();
  int i = 0;
  Iter state = cur;
  while (true) {
    for (int row = 0; row < kBoardDimension; ++row) {
      for (int col = 0; col < kBoardDimension; ++col) {
        tensor(2, row, col) =
          (state->aux.stable_discs & (1ULL << (row * kBoardDimension + col))) ? 1 : 0;
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

}  // namespace othello
