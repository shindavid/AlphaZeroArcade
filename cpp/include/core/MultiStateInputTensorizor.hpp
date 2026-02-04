#pragma once

#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/StaticCircularBuffer.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <core::concepts::Game Game, int NumPastStates>
class MultiStateInputTensorizorBase {
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

  void clear() { buf_.clear(); }
  void undo(const State&);
  void jump_to(StateIterator it);
  group::element_t get_random_symmetry() const;
  const State& current_state() const { return buf_.back().state; }
  void apply_action(const action_t action);
  void update(const State& state) { buf_.push_back({state, Symmetries::get_mask(state)}); }

 private:
  CircularBuffer buf_;
};

}  // namespace core

#include "inline/core/MultiStateInputTensorizor.inl"
