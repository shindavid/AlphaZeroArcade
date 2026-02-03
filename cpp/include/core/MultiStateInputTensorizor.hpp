#pragma once

#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/StaticCircularBuffer.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <core::concepts::Game Game, int NumPastStates>
class MultiStateInputTensorizor {
 public:
  using State = Game::State;
  using Rules = Game::Rules;
  using ActionMask = Game::Types::ActionMask;
  using StateIterator = core::StateIterator<Game>;
  using Symmetries = Game::Symmetries;
  using SymmetryMask = Game::Types::SymmetryMask;

  static constexpr int kNumStatesToEncode = NumPastStates + 1;  // +1 for current state
  static constexpr int kBufferSize = kNumStatesToEncode + 1;    // +1 for undo support

  struct Pair {
    State state;
    SymmetryMask sym_mask;
  };

  using CircularBuffer = util::StaticCircularBuffer<Pair, kBufferSize>;

  void clear() {}
  void update(const State& state) {};
  void undo(const State& state) {};
  void jump_to(StateIterator it) {};
  group::element_t get_random_symmetry() const;

 private:
  CircularBuffer buffer_;
};

}  // namespace core
