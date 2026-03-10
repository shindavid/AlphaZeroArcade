#pragma once

#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/FiniteGroups.hpp"
#include "util/StaticCircularBuffer.hpp"

namespace core {

template <core::concepts::Game Game, typename InputFrame, typename Symmetries, int NumPastStates>
class MultiStateInputTensorizorBase {
 public:
  using StateIterator = core::StateIterator<Game>;
  using SymmetryMask = Game::Types::SymmetryMask;
  using EvalKey = InputFrame;

  static_assert(NumPastStates > 0);
  static constexpr int kNumStatesToEncode = NumPastStates + 1;  // +1 for current state
  static constexpr int kBufferSize = kNumStatesToEncode + 1;    // +1 for undo support

  struct Pair {
    InputFrame frame;
    SymmetryMask sym_mask;
  };

  using CircularBuffer = util::StaticCircularBuffer<Pair, kBufferSize>;

  // size() returns the *logical* size of the buffer (excluding the extra slot reserved for undo
  // support).
  size_t size() const { return std::min(buf_.size(), static_cast<size_t>(kNumStatesToEncode)); }

  void clear();
  void undo();
  void jump_to(StateIterator it);
  group::element_t get_random_symmetry() const;
  group::element_t get_random_symmetry(const InputFrame& next_frame) const;
  const InputFrame& current_frame() const;
  void update(const InputFrame& frame);
  const CircularBuffer& buffer() const { return buf_; }
  EvalKey eval_key() const { return current_frame(); }

 private:
  CircularBuffer buf_;
  bool valid_ = false;
};

}  // namespace core

#include "inline/core/MultiStateInputTensorizor.inl"
