#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>

namespace generic {

/*
 * Abstract class. Derived classes must implement the prompt_for_action() method.
 */
template <core::concepts::Game Game>
class HumanTuiPlayer : public core::AbstractPlayer<Game> {
 public:
  using base_t = core::AbstractPlayer<Game>;

  using IO = Game::IO;
  using BaseState = Game::BaseState;
  using ActionMask = Game::Types::ActionMask;
  using ValueArray = Game::Types::ValueArray;
  using player_array_t = base_t::player_array_t;

  HumanTuiPlayer() {}
  virtual ~HumanTuiPlayer() {}
  void start_game() override;
  void receive_state_change(core::seat_index_t, const BaseState&, core::action_t) override;
  core::ActionResponse get_action_response(const BaseState&, const ActionMask&) override;
  void end_game(const BaseState&, const ValueArray&) override;

  bool is_human_tui_player() const override { return true; }

 protected:
  /*
   * Use std::cout/std::cin to prompt the user for an action. If an invalid action is entered, must
   * return -1.
   *
   * Derived classes must override this method.
   */
  virtual core::action_t prompt_for_action(const BaseState&, const ActionMask&) = 0;

  /*
   * By default, dispatches to Game::IO::dump(). Can be overridden by derived classes.
   */
  virtual void print_state(const BaseState&, bool terminal);

  core::action_t last_action_;
};

}  // namespace generic

#include <inline/generic_players/HumanTuiPlayer.inl>
