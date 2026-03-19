#include "core/SimpleInputTensorizor.hpp"

#include "util/Asserts.hpp"

namespace core {

template <core::concepts::Game Game, typename InputFrame, typename Symmetries>
group::element_t SimpleInputTensorizorBase<Game, InputFrame, Symmetries>::get_random_symmetry()
  const {
  return get_random_symmetry(current_frame());
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries>
group::element_t SimpleInputTensorizorBase<Game, InputFrame, Symmetries>::get_random_symmetry(
  const InputFrame& frame) {
  return Symmetries::get_mask(frame).choose_random_on_index();
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries>
const InputFrame& SimpleInputTensorizorBase<Game, InputFrame, Symmetries>::current_frame() const {
  return frame_;
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries>
void SimpleInputTensorizorBase<Game, InputFrame, Symmetries>::update(const State& state) {
  frame_ = state;
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries>
InputFrame SimpleInputTensorizorBase<Game, InputFrame, Symmetries>::eval_key() const {
  return frame_;
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries>
void SimpleInputTensorizorBase<Game, InputFrame, Symmetries>::restore(const InputFrame* frame,
                                                                      int num_frames) {
  DEBUG_ASSERT(num_frames == 1);
  update(*frame);
}

template <core::concepts::Game Game, typename InputFrame, typename Symmetries>
void SimpleInputTensorizorBase<Game, InputFrame, Symmetries>::apply_symmetry(group::element_t sym) {
  Symmetries::apply(frame_, sym);
}

}  // namespace core
