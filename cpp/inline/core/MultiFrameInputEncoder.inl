#include "core/MultiFrameInputEncoder.hpp"

#include "util/Asserts.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
void MultiFrameInputEncoderBase<Game, InputFrame, Symmetries, NumPastStates>::clear() {
  buf_.clear();
  valid_ = false;
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
void MultiFrameInputEncoderBase<Game, InputFrame, Symmetries, NumPastStates>::undo() {
  DEBUG_ASSERT(!buf_.empty());
  buf_.pop_back();
  valid_ = !buf_.empty();
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
void MultiFrameInputEncoderBase<Game, InputFrame, Symmetries, NumPastStates>::jump_to(
  InfoSetIterator it) {
  clear();
  while (buf_.size() < kNumFramesToEncode && !it.end()) {
    InputFrame frame(it->info_set);
    buf_.push_front({frame, Symmetries::get_mask(frame)});
    valid_ = true;
    ++it;
  }
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
group::element_t MultiFrameInputEncoderBase<Game, InputFrame, Symmetries,
                                            NumPastStates>::get_random_symmetry() const {
  DEBUG_ASSERT(valid_);
  auto begin = buf_.begin();
  auto end = buf_.end();
  int num_frames_to_use = std::min(static_cast<int>(buf_.size()), kNumFramesToEncode);
  auto it = std::max(begin, end - num_frames_to_use);

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
MultiFrameInputEncoderBase<Game, InputFrame, Symmetries, NumPastStates>::get_random_symmetry(
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
  int num_frames_to_use = std::min(static_cast<int>(buf_.size()), kNumFramesToEncode - 1);
  auto it = std::max(begin, end - num_frames_to_use);  // +1 to account for next state

  SymmetryMask mask = Symmetries::get_mask(next_frame);

  while (it != end) {
    mask &= it->sym_mask;
    ++it;
  }
  return mask.choose_random_on_index();
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
const InputFrame&
MultiFrameInputEncoderBase<Game, InputFrame, Symmetries, NumPastStates>::current_frame() const {
  DEBUG_ASSERT(!buf_.empty());
  DEBUG_ASSERT(valid_);
  return buf_.back().frame;
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
void MultiFrameInputEncoderBase<Game, InputFrame, Symmetries, NumPastStates>::update(
  const InputFrame& frame) {
  buf_.push_back({frame, Symmetries::get_mask(frame)});
  valid_ = true;
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
void MultiFrameInputEncoderBase<Game, InputFrame, Symmetries, NumPastStates>::restore(
  const InputFrame* frame, int num_frames) {
  for (int i = 0; i < num_frames; ++i) {
    update(frame[i]);
  }
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
void MultiFrameInputEncoderBase<Game, InputFrame, Symmetries, NumPastStates>::apply_symmetry(
  group::element_t sym) {
  DEBUG_ASSERT(valid_);
  for (auto& pair : buf_) {
    Symmetries::apply(pair.frame, sym);
  }
}

}  // namespace core
