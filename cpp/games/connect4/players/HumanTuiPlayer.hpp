#pragma once

#include <common/players/HumanTuiPlayer.hpp>
#include <games/connect4/GameState.hpp>
#include <games/connect4/players/PerfectPlayer.hpp>

namespace c4 {

class HumanTuiPlayer : public common::HumanTuiPlayer<GameState> {
public:
  using base_t = common::HumanTuiPlayer<GameState>;

  HumanTuiPlayer(bool cheat_mode);
  ~HumanTuiPlayer();

  void start_game() override;
  void receive_state_change(
      core::seat_index_t, const GameState&, const Action&) override;

private:
  Action prompt_for_action(const GameState&, const ActionMask&) override;
  int prompt_for_action_helper(const GameState&, const ActionMask&);
  void print_state(const GameState&, bool terminal) override;

  PerfectOracle* oracle_ = nullptr;
  PerfectOracle::MoveHistory* move_history_ = nullptr;
};

}  // namespace c4

#include <games/connect4/players/inl/HumanTuiPlayer.inl>
