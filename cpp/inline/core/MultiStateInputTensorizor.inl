#include "core/MultiStateInputTensorizor.hpp"
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
void MultiStateInputTensorizorBase<Game, NumPastStates>::apply_action(const action_t action) {
  DEBUG_ASSERT(!buf_.empty());
  State new_state = buf_.back().state;
  Rules::apply(new_state, action);
  buf_.push_back({new_state, Symmetries::get_mask(new_state)});
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
