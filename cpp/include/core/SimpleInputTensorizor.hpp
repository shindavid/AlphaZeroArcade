#pragma once

#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/FiniteGroups.hpp"

#include <type_traits>

namespace core {

template <core::concepts::Game Game, typename InputFrame, typename Symmetries>
class SimpleInputTensorizorBase {
 public:
  using StateIterator = core::StateIterator<Game>;
  using State = Game::State;
  static_assert(std::is_same_v<InputFrame, State>);
  using EvalKey = InputFrame;

  static constexpr int kNumFramesToEncode = 1;

  void clear() { valid_ = false; }
  void undo() { valid_ = false; }
  void jump_to(StateIterator it) { update(it->state); }
  group::element_t get_random_symmetry() const;
  static group::element_t get_random_symmetry(const InputFrame& frame);
  const InputFrame& current_frame() const;
  void update(const State& state);
  EvalKey eval_key() const;
  void restore(const InputFrame* frame, int num_frames);
  void apply_symmetry(group::element_t sym);

 private:
  InputFrame frame_;
  bool valid_ = false;
};

}  // namespace core

#include "inline/core/SimpleInputTensorizor.inl"
