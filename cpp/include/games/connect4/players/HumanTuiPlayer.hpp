#pragma once

#include <generic_players/HumanTuiPlayer.hpp>

#include <games/connect4/Game.hpp>
#include <games/connect4/players/PerfectPlayer.hpp>

namespace c4 {

class HumanTuiPlayer : public generic::HumanTuiPlayer<GameState> {
 public:
  using base_t = generic::HumanTuiPlayer<GameState>;

  HumanTuiPlayer(bool cheat_mode);
  ~HumanTuiPlayer();

  void start_game() override;
  void receive_state_change(core::seat_index_t, const GameState&, const Action&) override;

 private:
  Action prompt_for_action(const GameState&, const ActionMask&) override;
  int prompt_for_action_helper(const GameState&, const ActionMask&);
  void print_state(const GameState&, bool terminal) override;

  PerfectOracle* oracle_ = nullptr;
  PerfectOracle::MoveHistory* move_history_ = nullptr;
};

}  // namespace c4

#include <inline/games/connect4/players/HumanTuiPlayer.inl>
