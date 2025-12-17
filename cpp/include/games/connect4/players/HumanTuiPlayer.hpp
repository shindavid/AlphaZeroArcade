#pragma once

#include "games/connect4/Game.hpp"
#include "games/connect4/players/PerfectPlayer.hpp"
#include "generic_players/HumanTuiPlayer.hpp"

namespace c4 {

class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 public:
  using base_t = generic::HumanTuiPlayer<Game>;
  using State = Game::State;

  HumanTuiPlayer(bool cheat_mode);
  ~HumanTuiPlayer();

  bool start_game() override;
  void receive_state_change(core::seat_index_t, const State&, core::action_t,
                            core::node_ix_t) override;

 private:
  core::action_t prompt_for_action(const State&, const ActionMask&) override;
  void print_state(const State&, bool terminal) override;

  PerfectOracle* oracle_ = nullptr;
  PerfectOracle::MoveHistory* move_history_ = nullptr;
};

}  // namespace c4

#include "inline/games/connect4/players/HumanTuiPlayer.inl"
