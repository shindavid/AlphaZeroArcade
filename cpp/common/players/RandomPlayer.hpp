#pragma once

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <util/BitSet.hpp>

namespace common {

template<GameStateConcept GameState>
class RandomPlayer : public AbstractPlayer<GameState> {
public:
  using base_t = AbstractPlayer<GameState>;
  using GameStateTypes = common::GameStateTypes<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;

  action_index_t get_action(const GameState&, const ActionMask& mask) override {
    return bitset_util::choose_random_on_index(mask);
  }
};

}  // namespace common