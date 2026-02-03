#include "core/SimpleInputTensorizor.hpp"

namespace core {

template <core::concepts::Game Game>
group::element_t SimpleInputTensorizorBase<Game>::get_random_symmetry() const {
  SymmetryMask mask = Symmetries::get_mask(state_);
  return mask.choose_random_on_index();
}

template <core::concepts::Game Game>
void SimpleInputTensorizorBase<Game>::apply_action(const action_t action) {
  Rules::apply(state_, action);
}

}  // namespace core
