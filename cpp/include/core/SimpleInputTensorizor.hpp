#pragma once

#include "core/DefaultKeys.hpp"
#include "core/StateIterator.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/FiniteGroups.hpp"

namespace core {

template <core::concepts::Game Game, typename Symmetries>
class SimpleInputTensorizorBase {
 public:
  using Keys = core::DefaultKeys<Game>;
  using State = Game::State;
  using Unit = State;
  using Rules = Game::Rules;
  using ActionMask = Game::Types::ActionMask;
  using StateIterator = core::StateIterator<Game>;
  using SymmetryMask = Game::Types::SymmetryMask;

  static constexpr int kNumStatesToEncode = 1;

  void clear() { valid_ = false; }
  void undo() { valid_ = false; }
  void jump_to(StateIterator it) { update(it->state); }
  group::element_t get_random_symmetry() const {
    RELEASE_ASSERT(valid_);
    return get_random_symmetry(state_);
  }
  static group::element_t get_random_symmetry(const State& state) {
    return Symmetries::get_mask(state).choose_random_on_index();
  }
  const Unit& current_unit() const {
    RELEASE_ASSERT(valid_);
    return state_;
  }
  void update(const State& state) {
    state_ = state;
    valid_ = true;
  }

 protected:
  const State& state() const { return state_; }

 private:
  State state_;
  bool valid_ = false;
};

}  // namespace core
