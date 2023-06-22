#pragma once

#include <core/players/HumanTuiPlayer.hpp>
#include <connect4/GameState.hpp>
#include <connect4/players/PerfectPlayer.hpp>

namespace c4 {

class HumanTuiPlayer : public common::HumanTuiPlayer<GameState> {
public:
  using base_t = common::HumanTuiPlayer<GameState>;

  HumanTuiPlayer(bool cheat_mode);
  ~HumanTuiPlayer();

  void start_game() override;
  void receive_state_change(
      common::seat_index_t, const GameState&, common::action_index_t) override;

private:
  common::action_index_t prompt_for_action(const GameState&, const ActionMask&) override;
  void print_state(const GameState&, bool terminal) override;

  PerfectOracle* oracle_ = nullptr;
  PerfectOracle::MoveHistory* move_history_ = nullptr;
};

}  // namespace c4

#include <connect4/players/inl/HumanTuiPlayer.inl>
