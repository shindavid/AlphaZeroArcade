#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/concepts/Game.hpp>

namespace generic {

/*
 * Abstract class. Derived classes must implement the prompt_for_action() method.
 */
template <core::concepts::Game Game_>
class HumanTuiPlayer : public core::AbstractPlayer<Game_> {
 public:
  using Game = Game_;
  using base_t = core::AbstractPlayer<Game>;

  using IO = typename Game::IO;
  using FullState = typename Game::FullState;
  using ActionMask = typename Game::ActionMask;
  using ValueArray = typename Game::ValueArray;
  using player_array_t = typename base_t::player_array_t;

  HumanTuiPlayer() {}
  virtual ~HumanTuiPlayer() {}
  void start_game() override;
  void receive_state_change(core::seat_index_t, const FullState&, core::action_t) override;
  ActionResponse get_action_response(const FullState&, const ActionMask&) override;
  void end_game(const FullState&, const ValueArray&) override;

  bool is_human_tui_player() const override { return true; }

 protected:
  /*
   * Use std::cout/std::cin to prompt the user for an action. If an invalid action is entered, must
   * return -1.
   *
   * Derived classes must override this method.
   */
  virtual core::action_t prompt_for_action(const FullState&, const ActionMask&) = 0;

  /*
   * By default, dispatches to Game::IO::dump(). Can be overridden by derived classes.
   */
  virtual void print_state(const FullState&, bool terminal);

  core::action_t last_action_;
};

}  // namespace generic

#include <inline/generic_players/HumanTuiPlayer.inl>
