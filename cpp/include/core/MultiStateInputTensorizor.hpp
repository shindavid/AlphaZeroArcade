#pragma once

#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/StaticCircularBuffer.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <typename UnitBuilder, core::concepts::Game Game, int NumPastStates>
class MultiStateInputTensorizorBase {
 public:
  using State = Game::State;
  using Unit = UnitBuilder::Unit;
  using StateIterator = core::StateIterator<Game>;
  using Symmetries = Game::Symmetries;
  using SymmetryMask = Game::Types::SymmetryMask;

  static_assert(NumPastStates > 0);
  static constexpr int kNumStatesToEncode = NumPastStates + 1;  // +1 for current state
  static constexpr int kBufferSize = kNumStatesToEncode + 1;    // +1 for undo support

  struct Pair {
    Unit unit;
    SymmetryMask sym_mask;
  };

  using CircularBuffer = util::StaticCircularBuffer<Pair, kBufferSize>;

  // size() returns the *logical* size of the buffer (excluding the extra slot reserved for undo
  // support).
  size_t size() const { return std::min(buf_.size(), static_cast<size_t>(kNumStatesToEncode)); }

  void clear() { buf_.clear(); valid_ = false; }
  void undo();
  void jump_to(StateIterator it);
  group::element_t get_random_symmetry() const;
  group::element_t get_random_symmetry(const State& next_state) const;
  const Unit& current_unit() const;
  void update(const State& state);
  const CircularBuffer& buffer() const { return buf_; }

 private:
  CircularBuffer buf_;
  bool valid_ = false;
};

}  // namespace core

#include "inline/core/MultiStateInputTensorizor.inl"
