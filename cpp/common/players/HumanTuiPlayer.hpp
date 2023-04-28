#pragma once

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>

namespace common {

template<GameStateConcept GameState_>
class HumanTuiPlayer : public AbstractPlayer<GameState_> {
public:
  using base_t = AbstractPlayer<GameState_>;
  using GameState = GameState_;
  using GameStateTypes = common::GameStateTypes<GameState>;

  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using player_array_t = typename base_t::player_array_t;

  HumanTuiPlayer() {}
  void start_game() override;
  void receive_state_change(common::seat_index_t, const GameState&, common::action_index_t) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;
  void end_game(const GameState&, const GameOutcome&) override;

  bool is_human_tui_player() const override { return true; }

protected:
  virtual void print_state(const GameState&, bool terminal);

  common::action_index_t last_action_ = -1;
};

}  // namespace common

#include <common/players/inl/HumanTuiPlayer.inl>

