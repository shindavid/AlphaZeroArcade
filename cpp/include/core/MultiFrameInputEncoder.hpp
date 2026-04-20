#pragma once

#include "core/InfoSetIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/FiniteGroups.hpp"
#include "util/StaticCircularBuffer.hpp"

namespace core {

template <core::concepts::Game Game_, typename InputFrame_, typename Symmetries, int NumPastStates>
class MultiFrameInputEncoderBase {
 public:
  using Game = Game_;
  using InputFrame = InputFrame_;
  using InfoSetIterator = core::InfoSetIterator<Game>;
  using SymmetryMask = Game::Types::SymmetryMask;

  static_assert(NumPastStates > 0);
  static constexpr int kNumFramesToEncode = NumPastStates + 1;  // +1 for current state
  static constexpr int kBufferSize = kNumFramesToEncode + 1;    // +1 for undo support

  struct Pair {
    InputFrame frame;
    SymmetryMask sym_mask;
  };

  using CircularBuffer = util::StaticCircularBuffer<Pair, kBufferSize>;

  // size() returns the *logical* size of the buffer (excluding the extra slot reserved for undo
  // support).
  size_t size() const { return std::min(buf_.size(), static_cast<size_t>(kNumFramesToEncode)); }

  void clear();
  void undo();
  void jump_to(InfoSetIterator it);
  group::element_t get_random_symmetry() const;
  group::element_t get_random_symmetry(const InputFrame& next_frame) const;
  const InputFrame& current_frame() const;
  void update(const InputFrame& frame);
  const CircularBuffer& buffer() const { return buf_; }
  void restore(const InputFrame* frame, int num_frames);
  void apply_symmetry(group::element_t sym);

 private:
  CircularBuffer buf_;
  bool valid_ = false;
};

}  // namespace core

#include "inline/core/MultiFrameInputEncoder.inl"
