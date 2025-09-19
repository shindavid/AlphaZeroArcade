#include "games/othello/Bindings.hpp"

namespace othello::alpha0 {

template <typename GameLogView>
inline bool TrainingTargets::ScoreMarginTarget::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor.setZero();
  const Game::State& state = view.final_pos;
  core::seat_index_t cp = Game::Rules::get_current_player(view.cur_pos);
  int score_index = kNumCells + state.get_count(cp) - state.get_count(1 - cp);
  RELEASE_ASSERT(score_index >= 0 && score_index <= kNumCells * 2);

  // PDF
  tensor(0, score_index) = 1;

  // CDF
  for (int i = 0; i <= score_index; ++i) {
    tensor(1, i) = 1;
  }
  return true;
}

template <typename GameLogView>
inline bool TrainingTargets::OwnershipTarget::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor.setZero();
  core::seat_index_t cp = Game::Rules::get_current_player(view.cur_pos);
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      core::seat_index_t p = view.final_pos.get_player_at(row, col);
      int x = (p == -1) ? 2 : ((p == cp) ? 0 : 1);
      tensor(x, row, col) = 1;
    }
  }

  return true;
}

}  // namespace othello::alpha0
