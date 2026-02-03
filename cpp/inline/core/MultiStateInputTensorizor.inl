#include "core/MultiStateInputTensorizor.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <core::concepts::Game Game, int NumPastStates>
group::element_t MultiStateInputTensorizor<Game, NumPastStates>::get_random_symmetry() const {
  auto it = buffer_.begin();
  SymmetryMask mask = Symmetries::get_mask(*it);
  it++;
  while (it != buffer_.end()) {
    mask &= Symmetries::get_mask(*it);
    ++it;
  }
  return mask.choose_random_on_index();
}

}  // namespace core
