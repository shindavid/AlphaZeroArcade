#include "core/MultiStateInputTensorizor.hpp"

#include "util/Asserts.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
group::element_t MultiStateInputTensorizorBase<Game, InputFrame, Symmetries,
                                               NumPastStates>::get_random_symmetry() const {
  DEBUG_ASSERT(valid_);
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

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
group::element_t
MultiStateInputTensorizorBase<Game, InputFrame, Symmetries, NumPastStates>::get_random_symmetry(
  const InputFrame& next_frame) const {
  // Simulate the following:
  //
  // update(next_frame);
  // auto sym = gen_random_symmetry();
  // undo();
  // return sym;

  DEBUG_ASSERT(valid_);
  auto begin = buf_.begin();
  auto end = buf_.end();
  auto it = std::max(begin, end - kNumStatesToEncode + 1);  // +1 to account for next state

  SymmetryMask mask = Symmetries::get_mask(next_frame);
  ;
  while (it != end) {
    mask &= it->sym_mask;
    ++it;
  }
  return mask.choose_random_on_index();
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
void MultiStateInputTensorizorBase<Game, InputFrame, Symmetries, NumPastStates>::clear() {
  buf_.clear();
  valid_ = false;
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
void MultiStateInputTensorizorBase<Game, InputFrame, Symmetries, NumPastStates>::undo() {
  DEBUG_ASSERT(!buf_.empty());
  buf_.pop_back();
  valid_ = false;
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
const InputFrame&
MultiStateInputTensorizorBase<Game, InputFrame, Symmetries, NumPastStates>::current_frame() const {
  DEBUG_ASSERT(!buf_.empty());
  DEBUG_ASSERT(valid_);
  return buf_.back().frame;
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
void MultiStateInputTensorizorBase<Game, InputFrame, Symmetries, NumPastStates>::update(
  const InputFrame& frame) {
  buf_.push_back({frame, Symmetries::get_mask(frame)});
  valid_ = true;
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
void MultiStateInputTensorizorBase<Game, InputFrame, Symmetries, NumPastStates>::jump_to(
  StateIterator it) {
  clear();
  while (buf_.size() < kNumStatesToEncode && !it.end()) {
    InputFrame frame(it->state);
    buf_.push_front({frame, Symmetries::get_mask(frame)});
    valid_ = true;
    ++it;
  }
}

}  // namespace core
