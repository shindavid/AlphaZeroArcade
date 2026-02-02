#include "core/InputTensorizor.hpp"

namespace core {

template <core::concepts::Game Game>
void SimpleInputTensorizorBase<Game>::update(const State& state) {
  state_ = state;
  mask_ = Rules::get_legal_actions(state);
}

template <core::concepts::Game Game>
group::element_t SimpleInputTensorizorBase<Game>::get_random_symmetry() {
  SymmetryMask mask = Symmetries::get_mask(state_);
  return mask.choose_random_on_index();
}

}  // namespace core
