#pragma once

#include <generic_players/HumanTuiPlayer.hpp>

#include <games/connect4/Game.hpp>
#include <games/connect4/players/PerfectPlayer.hpp>

namespace c4 {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 public:
  using base_t = generic::HumanTuiPlayer<Game>;
  using FullState = Game::FullState;

  HumanTuiPlayer(bool cheat_mode);
  ~HumanTuiPlayer();

  void start_game() override;
  void receive_state_change(core::seat_index_t, const FullState&, core::action_t) override;

 private:
  core::action_t prompt_for_action(const FullState&, const ActionMask&) override;
  void print_state(const FullState&, bool terminal) override;

  PerfectOracle* oracle_ = nullptr;
  PerfectOracle::MoveHistory* move_history_ = nullptr;
};

}  // namespace c4

#include <inline/games/connect4/players/HumanTuiPlayer.inl>
