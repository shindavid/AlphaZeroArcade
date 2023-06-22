#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>

namespace common {

/*
 * Abstract class. Derived classes must implement the prompt_for_action() method.
 */
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
  virtual ~HumanTuiPlayer() {}
  void start_game() override;
  void receive_state_change(common::seat_index_t, const GameState&, common::action_index_t) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;
  void end_game(const GameState&, const GameOutcome&) override;

  bool is_human_tui_player() const override { return true; }

protected:
  /*
   * Use std::cout/std::cin to prompt the user for an action. If an invalid action is entered, must return -1.
   *
   * Derived classes must override this method.
   */
  virtual action_index_t prompt_for_action(const GameState&, const ActionMask&) = 0;

  /*
   * By default, dispatches to GameState::dump(). Can be overridden by derived classes.
   */
  virtual void print_state(const GameState&, bool terminal);

  common::action_index_t last_action_ = -1;
};

}  // namespace common

#include <core/players/inl/HumanTuiPlayer.inl>

