#include "games/blokus/Bindings.hpp"

namespace blokus::alpha0 {

inline bool TrainingTargets::ScoreTarget::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor.setZero();
  const Game::State& state = *view.final_pos;
  color_t cp = Game::Rules::get_current_player(*view.cur_pos);

  int scores[kNumColors];
  for (color_t c = 0; c < kNumColors; ++c) {
    int score = state.remaining_square_count(c);
    scores[c] = std::min(score, kVeryBadScore);
  }

  for (color_t c = 0; c < kNumColors; ++c) {
    color_t rc = (kNumColors + c - cp) % kNumColors;

    // PDF
    tensor(0, scores[c], rc) = 1;

    // CDF
    for (int score = 0; score <= scores[c]; ++score) {
      tensor(1, score, rc) = 1;
    }
  }

  return true;
}

inline bool TrainingTargets::OwnershipTarget::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor.setZero();
  const Game::State& state = *view.final_pos;
  color_t cp = Game::Rules::get_current_player(*view.cur_pos);

  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      tensor(kNumColors, row, col) = 1;
    }
  }

  for (color_t c = 0; c < kNumColors; ++c) {
    color_t rc = (kNumColors + c - cp) % kNumColors;
    for (Location loc : state.core.occupied_locations[c].get_set_locations()) {
      tensor(rc, loc.row, loc.col) = 1;
      tensor(kNumColors, loc.row, loc.col) = 0;
    }
  }

  return true;
}

inline bool TrainingTargets::UnplayedPiecesTarget::tensorize(const GameLogView& view,
                                                             Tensor& tensor) {
  tensor.setZero();
  const Game::State& state = *view.final_pos;
  color_t cp = Game::Rules::get_current_player(*view.cur_pos);

  for (color_t c = 0; c < kNumColors; ++c) {
    const PieceMask& mask = state.aux.played_pieces[c];
    color_t rc = (kNumColors + c - cp) % kNumColors;
    for (auto p : mask.get_unset_bits()) {
      tensor(rc, p) = 1;
    }
  }

  return true;
}

}  // namespace blokus::alpha0
