#pragma once

#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <core::concepts::Game Game, typename InputFrame, typename Symmetries>
class SimpleInputTensorizorBase {
 public:
  using StateIterator = core::StateIterator<Game>;
  using EvalKey = InputFrame;

  static constexpr int kNumFramesToEncode = 1;

  void clear() {}
  void undo() {}
  void jump_to(StateIterator it) { update(it->state); }
  group::element_t get_random_symmetry() const;
  static group::element_t get_random_symmetry(const InputFrame& frame);
  const InputFrame& current_frame() const;
  void update(const InputFrame& frame);
  EvalKey eval_key() const;
  void restore(const InputFrame* frame, int num_frames);
  void apply_symmetry(group::element_t sym);

 private:
  InputFrame frame_;
};

}  // namespace core

#include "inline/core/SimpleInputTensorizor.inl"
