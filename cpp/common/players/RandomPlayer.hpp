#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <util/BitSet.hpp>

namespace common {

/*
 * RandomPlayer always chooses uniformly at random among the set of legal moves.
 */
template<core::GameStateConcept GameState>
class RandomPlayer : public core::AbstractPlayer<GameState> {
public:
  using base_t = core::AbstractPlayer<GameState>;
  using GameStateTypes = core::GameStateTypes<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;

  core::action_t get_action(const GameState&, const ActionMask& mask) override {
    return bitset_util::choose_random_on_index(mask);
  }
};

}  // namespace common
