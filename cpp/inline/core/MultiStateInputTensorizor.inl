#include "core/MultiStateInputTensorizor.hpp"

#include "util/Asserts.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <typename UnitBuilder, core::concepts::Game Game, int NumPastStates>
group::element_t
MultiStateInputTensorizorBase<UnitBuilder, Game, NumPastStates>::get_random_symmetry() const {
  RELEASE_ASSERT(valid_);
  auto begin = buf_.begin();
  auto end = buf_.end();
  auto it = std::max(begin, end - kNumStatesToEncode);

  SymmetryMask mask = it->sym_mask;
  it++;
  while (it != end) {
    mask &= it->sym_mask;
    ++it;
  }
  return mask.choose_random_on_index();
}

template <typename UnitBuilder, core::concepts::Game Game, int NumPastStates>
group::element_t
MultiStateInputTensorizorBase<UnitBuilder, Game, NumPastStates>::get_random_symmetry(
  const State& next_state) const {
  // Simulate the following:
  //
  // update(next_state);
  // auto sym = gen_random_symmetry();
  // undo();
  // return sym;

  RELEASE_ASSERT(valid_);
  auto begin = buf_.begin();
  auto end = buf_.end();
  auto it = std::max(begin, end - kNumStatesToEncode + 1);  // +1 to account for next state

  SymmetryMask mask = Symmetries::get_mask(next_state);
  ;
  while (it != end) {
    mask &= it->sym_mask;
    ++it;
  }
  return mask.choose_random_on_index();
}

template <typename UnitBuilder, core::concepts::Game Game, int NumPastStates>
void MultiStateInputTensorizorBase<UnitBuilder, Game, NumPastStates>::undo() {
  DEBUG_ASSERT(!buf_.empty());
  buf_.pop_back();
  valid_ = false;
}

template <typename UnitBuilder, core::concepts::Game Game, int NumPastStates>
const typename MultiStateInputTensorizorBase<UnitBuilder, Game, NumPastStates>::Unit&
MultiStateInputTensorizorBase<UnitBuilder, Game, NumPastStates>::current_unit() const {
  RELEASE_ASSERT(!buf_.empty());
  RELEASE_ASSERT(valid_);
  return buf_.back().unit;
}

template <typename UnitBuilder, core::concepts::Game Game, int NumPastStates>
void MultiStateInputTensorizorBase<UnitBuilder, Game, NumPastStates>::update(const State& state) {
  buf_.push_back({UnitBuilder::build(state), Symmetries::get_mask(state)});
  valid_ = true;
}

template <typename UnitBuilder, core::concepts::Game Game, int NumPastStates>
void MultiStateInputTensorizorBase<UnitBuilder, Game, NumPastStates>::jump_to(StateIterator it) {
  clear();
  while (buf_.size() < kNumStatesToEncode && !it.end()) {
    buf_.push_front({UnitBuilder::build(it->state), Symmetries::get_mask(it->state)});
    valid_ = true;
    ++it;
  }
}

}  // namespace core
