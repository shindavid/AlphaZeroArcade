#include "games/tictactoe/Bindings.hpp"

namespace tictactoe::alpha0 {

inline bool TrainingTargets::OwnershipTarget::tensorize(const Game::Types::GameLogView& view,
                                                        Tensor& tensor) {
  tensor.setZero();
  const Game::State& state = *view.final_pos;
  core::seat_index_t cp = Game::Rules::get_current_player(*view.cur_pos);
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t p = state.get_player_at(row, col);
      int x = (p == -1) ? 2 : ((p == cp) ? 0 : 1);
      tensor(x, row, col) = 1;
    }
  }
  return true;
}

}  // namespace tictactoe::alpha0
