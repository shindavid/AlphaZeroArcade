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
  using player_array_t = typename base_t::player_array_t;
  using GameStateTypes = common::GameStateTypes<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;

  RandomPlayer() : base_t("Random") {}

  void start_game(game_id_t, const player_array_t& players, player_index_t seat_assignment) override {}
  void receive_state_change(player_index_t, const GameState&, action_index_t, const GameOutcome&) override {}
  action_index_t get_action(const GameState&, const ActionMask& mask) override {
    return bitset_util::choose_random_on_index(mask);
  }
};

}  // namespace common
