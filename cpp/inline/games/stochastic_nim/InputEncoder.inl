#include "games/stochastic_nim/InputEncoder.hpp"

namespace stochastic_nim {

inline InputEncoder::Tensor InputEncoder::encode(group::element_t sym) {
  InputFrame frame = this->current_frame();
  Symmetries::apply(frame, sym);

  Tensor tensor;
  tensor.setZero();
  for (int i = 0; i < kStartingStonesBitWidth; ++i) {
    tensor(0, i, 0) = (frame.stones_left & (1 << i)) ? 1 : 0;
  }
  tensor(0, kStartingStonesBitWidth, 0) = frame.current_phase;
  return tensor;
}

}  // namespace stochastic_nim
