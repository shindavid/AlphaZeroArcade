#include "games/nim/InputEncoder.hpp"

namespace nim {

inline InputEncoder::Tensor InputEncoder::encode(group::element_t sym) {
  InputFrame frame = this->current_frame();
  Symmetries::apply(frame, sym);

  Tensor tensor;
  tensor.setZero();
  for (int i = 0; i < frame.stones_left; ++i) {
    tensor(nim::kStartingStones - 1 - i) = 1;
  }
  return tensor;
}

}  // namespace nim
