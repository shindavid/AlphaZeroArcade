#pragma once

#include <ostream>

#include <common/Types.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>

namespace c4 {

class HumanTuiPlayer : public Player {
public:
  using base_t = common::AbstractPlayer<GameState>;

  HumanTuiPlayer() : base_t("Human") {}
  void start_game(const player_array_t& players, common::player_index_t seat_assignment) override;
  void receive_state_change(common::player_index_t, const GameState&, common::action_index_t, const Result&) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;

private:
  void xprintf_switch(const GameState&);
  void print_state(const GameState&);

  std::ostringstream buf_;
  player_name_array_t player_names_;
  common::player_index_t my_index_ = -1;
  common::action_index_t last_action_ = -1;
};

}  // namespace c4

#include <connect4/inl/C4HumanTuiPlayer.inl>
