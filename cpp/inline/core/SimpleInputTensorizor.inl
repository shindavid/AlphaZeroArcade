#include "core/SimpleInputTensorizor.hpp"

namespace core {

template <core::concepts::Game Game>
group::element_t SimpleInputTensorizorBase<Game>::get_random_symmetry() const {
  SymmetryMask mask = Symmetries::get_mask(state_);
  return mask.choose_random_on_index();
}

}  // namespace core
