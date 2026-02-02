#pragma once

#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/FiniteGroups.hpp"

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
  using Symmetries = Game::Symmetries;
  using SymmetryMask = Game::Types::SymmetryMask;

  static constexpr int kNumStatesToEncode = 1;

  void clear() {}
  void update(const State& state);
  void undo(const State& state) { update(state); }
  void jump_to(StateIterator it) { update(it->state); }
  group::element_t get_random_symmetry();

 protected:
  const State& state() const { return state_; }
  const ActionMask& valid_actions() const { return mask_; }

 private:
  State state_;
  ActionMask mask_;
};

}  // namespace core

#include "inline/core/InputTensorizor.inl"
