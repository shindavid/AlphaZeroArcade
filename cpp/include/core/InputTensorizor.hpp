#pragma once

#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <core::concepts::Game Game>
struct InputTensorizor;  // no definition: require a specialization per game

template <core::concepts::Game Game>
class SimpleInputTensorizorBase {
 public:
  using State = Game::State;
  using Rules = Game::Rules;
  using ActionMask = Game::Types::ActionMask;
  using StateIterator = core::StateIterator<Game>;

  static constexpr int kNumStatesToEncode = 1;

  void clear() {}
  void update(const State& state);
  void undo(const State& state) { update(state); }
  void jump_to(StateIterator it) { update(it->state); }

 protected:
  const State& state() const { return state_; }
  const ActionMask& valid_actions() const { return mask_; }

 private:
  State state_;
  ActionMask mask_;
};

}  // namespace core

#include "inline/core/InputTensorizor.inl"
