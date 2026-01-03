#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/BasicTypes.hpp"
#include "core/StateChangeUpdate.hpp"
#include "core/TreePanel.hpp"
#include "core/concepts/GameConcept.hpp"

namespace generic {

/*
 * Abstract class. Derived classes must implement the prompt_for_action() method.
 */
template <core::concepts::Game Game>
class HumanTuiPlayer : public core::AbstractPlayer<Game> {
 public:
  using base_t = core::AbstractPlayer<Game>;

  using IO = Game::IO;
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = core::ActionRequest<Game>;
  using GameResultTensor = Game::Types::GameResultTensor;
  using StateChangeUpdate = core::StateChangeUpdate<Game>;
  using player_array_t = base_t::player_array_t;

  HumanTuiPlayer() {}
  virtual ~HumanTuiPlayer() {}
  bool start_game() override;
  void receive_state_change(const StateChangeUpdate&) override;
  core::ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State&, const GameResultTensor&) override;

  bool disable_progress_bar() const override { return true; }
  void backtrack_to_node(core::game_tree_index_t ix) override { active_node_index_ = ix; }

 protected:
  /*
   * Use std::cout/std::cin to prompt the user for an action. If an invalid action is entered, must
   * return -1.
   *
   * Derived classes must override this method.
   */
  virtual core::ActionResponse prompt_for_action(const ActionRequest&) = 0;

  /*
   * By default, dispatches to Game::IO::dump(). Can be overridden by derived classes.
   */
  virtual void print_state(const State&, bool terminal);

  core::action_t last_action_;

 private:
  core::TreePanel* tree_panel_ = core::TreePanel::get_instance();
  core::game_tree_index_t active_node_index_ = 0;
};

}  // namespace generic

#include "inline/generic_players/HumanTuiPlayer.inl"
