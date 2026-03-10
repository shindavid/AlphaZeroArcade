#pragma once

#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/Asserts.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <core::concepts::Game Game, typename InputFrame, typename Symmetries>
class SimpleInputTensorizorBase {
 public:
  using StateIterator = core::StateIterator<Game>;
  using EvalKey = InputFrame;

  static constexpr int kNumStatesToEncode = 1;

  void clear() { valid_ = false; }
  void undo() { valid_ = false; }
  void jump_to(StateIterator it) { update(it->state); }
  group::element_t get_random_symmetry() const {
    return get_random_symmetry(current_frame());
  }
  static group::element_t get_random_symmetry(const InputFrame& frame) {
    return Symmetries::get_mask(frame).choose_random_on_index();
  }
  const InputFrame& current_frame() const {
    DEBUG_ASSERT(valid_);
    return frame_;
  }
  void update(const InputFrame& frame) {
    frame_ = frame;
    valid_ = true;
  }

  EvalKey eval_key() const {
    DEBUG_ASSERT(valid_);
    return frame_;
  }

 private:
  InputFrame frame_;
  bool valid_ = false;
};

}  // namespace core
