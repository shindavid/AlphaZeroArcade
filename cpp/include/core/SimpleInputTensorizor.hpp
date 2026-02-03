#pragma once

#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <core::concepts::Game Game>
class SimpleInputTensorizorBase {
 public:
  using State = Game::State;
  using Rules = Game::Rules;
  using ActionMask = Game::Types::ActionMask;
  using StateIterator = core::StateIterator<Game>;
  using Symmetries = Game::Symmetries;
  using SymmetryMask = Game::Types::SymmetryMask;

  static constexpr int kNumStatesToEncode = 1;

  void clear() {}
  void undo(const State& state) { update(state); }
  void jump_to(StateIterator it) { update(it->state); }
  group::element_t get_random_symmetry() const;
  const State& current_state() const { return state_; }
  void apply_action(const action_t action);
  void update(const State& state) { state_ = state; }

 protected:
  const State& state() const { return state_; }

 private:
  State state_;
};

}  // namespace core

#include "inline/core/SimpleInputTensorizor.inl"
