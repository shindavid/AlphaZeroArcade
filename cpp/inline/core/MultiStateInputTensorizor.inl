#include "core/MultiStateInputTensorizor.hpp"
#include "util/Asserts.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <core::concepts::Game Game, int NumPastStates>
group::element_t MultiStateInputTensorizorBase<Game, NumPastStates>::get_random_symmetry() const {
  auto it = buf_.begin();
  SymmetryMask mask = it->sym_mask;
  it++;
  while (it != buf_.end()) {
    mask &= it->sym_mask;
    ++it;
  }
  return mask.choose_random_on_index();
}

template <core::concepts::Game Game, int NumPastStates>
void MultiStateInputTensorizorBase<Game, NumPastStates>::undo(const State&) {
  DEBUG_ASSERT(!buf_.empty());
  buf_.pop_back();
}

template <core::concepts::Game Game, int NumPastStates>
const MultiStateInputTensorizorBase<Game, NumPastStates>::State&
MultiStateInputTensorizorBase<Game, NumPastStates>::current_state() const {
  RELEASE_ASSERT(!buf_.empty());
  return buf_.back().state;
}

template <core::concepts::Game Game, int NumPastStates>
void MultiStateInputTensorizorBase<Game, NumPastStates>::jump_to(StateIterator it) {
  buf_.clear();
  while (buf_.size() < kNumStatesToEncode && !it.end()) {
    buf_.push_front({it->state, Symmetries::get_mask(it->state)});
    ++it;
  }
}

}  // namespace core
