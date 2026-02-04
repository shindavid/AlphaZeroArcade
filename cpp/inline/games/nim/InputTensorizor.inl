#include "games/nim/InputTensorizor.hpp"

namespace nim {

inline InputTensorizor::Tensor InputTensorizor::tensorize(group::element_t sym) {
  State state = this->state();

  Tensor tensor;
  tensor.setZero();
  for (int i = 0; i < state.stones_left; ++i) {
    tensor(nim::kStartingStones - 1 - i) = 1;
  }
  return tensor;
}

}  // namespace nim
