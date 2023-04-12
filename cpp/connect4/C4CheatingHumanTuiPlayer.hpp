#pragma once

#include <common/HumanTuiPlayer.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>

namespace c4 {

class CheatingHumanTuiPlayer : public common::HumanTuiPlayer<GameState> {
public:
  using base_t = common::HumanTuiPlayer<GameState>;

  void start_game() override;
  void receive_state_change(
      common::player_index_t, const GameState&, common::action_index_t) override;

private:
  void print_state(const GameState&, bool terminal) override;

  PerfectOracle oracle_;
  PerfectOracle::MoveHistory move_history_;
};

}  // namespace c4

#include <connect4/inl/C4CheatingHumanTuiPlayer.inl>
