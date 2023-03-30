#pragma once

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/HumanTuiPlayerBase.hpp>

namespace common {

template<GameStateConcept GameState_>
class HumanTuiPlayer : public AbstractPlayer<GameState_>, public HumanTuiPlayerBase {
public:
  using base_t = AbstractPlayer<GameState_>;
  using GameState = GameState_;
  using GameStateTypes = common::GameStateTypes<GameState>;

  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using player_name_array_t = typename GameStateTypes::player_name_array_t;
  using player_array_t = typename base_t::player_array_t;

  HumanTuiPlayer() : base_t("Human") {}
  void start_game() override;
  void receive_state_change(common::player_index_t, const GameState&, common::action_index_t, const GameOutcome&) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;
  void disable_screen_clearing() { screen_clearing_enabled_ = false; }

protected:
  virtual void print_state(const GameState&);

  player_name_array_t player_names_;
  common::action_index_t last_action_ = -1;
  bool screen_clearing_enabled_ = true;
};

}  // namespace common

#include <common/inl/HumanTuiPlayer.inl>
