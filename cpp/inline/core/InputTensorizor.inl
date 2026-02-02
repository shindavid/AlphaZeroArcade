#include "core/InputTensorizor.hpp"

namespace core {

template <core::concepts::Game Game>
void SimpleInputTensorizorBase<Game>::update(const State& state) {
  state_ = state;
  mask_ = Rules::get_legal_actions(state);
}

}  // namespace core
