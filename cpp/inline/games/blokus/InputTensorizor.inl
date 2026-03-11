#include "games/blokus/InputTensorizor.hpp"

namespace blokus {

inline InputTensorizor::Tensor InputTensorizor::tensorize(group::element_t sym) {
  InputFrame frame = this->current_frame();
  Symmetries::apply(frame, sym);
  core::seat_index_t cp = Game::Rules::get_current_player(frame);

  Tensor tensor;
  tensor.setZero();

  for (color_t c = 0; c < kNumColors; ++c) {
    color_t rc = (kNumColors + c - cp) % kNumColors;
    for (Location loc : frame.core.occupied_locations[c].get_set_locations()) {
      tensor(rc, loc.row, loc.col) = 1;
    }
  }

  if (frame.core.partial_move.valid()) {
    Location loc = frame.core.partial_move;
    tensor(kDim0 - 1, loc.row, loc.col) = 1;
  }
  return tensor;
}

}  // namespace blokus
