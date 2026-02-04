#include "games/stochastic_nim/InputTensorizor.hpp"

namespace stochastic_nim {

inline InputTensorizor::Tensor InputTensorizor::tensorize(group::element_t sym) {
  State state = this->state();
  Symmetries::apply(state, sym);

  Tensor tensor;
  tensor.setZero();
  for (int i = 0; i < kStartingStonesBitWidth; ++i) {
    tensor(0, i, 0) = (state.stones_left & (1 << i)) ? 1 : 0;
  }
  tensor(0, kStartingStonesBitWidth, 0) = state.current_mode;
  return tensor;
}

}  // namespace stochastic_nim
