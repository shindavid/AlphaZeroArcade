#include <common/HumanTuiPlayer.hpp>

#include <cstdlib>
#include <iostream>

#include <common/DerivedTypes.hpp>
#include <util/ScreenUtil.hpp>

namespace common {

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::start_game(
    game_id_t, const player_array_t& players, common::player_index_t seat_assignment)
{
  for (int p = 0; p < int(player_names_.size()); ++p) {
    player_names_[p] = players[p]->get_name();
  }
  my_index_ = seat_assignment;
  util::clearscreen();
}

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::receive_state_change(
    common::player_index_t, const GameState& state, common::action_index_t action, const GameOutcome& outcome)
{
  last_action_ = action;
}

template<GameStateConcept GameState_>
inline common::action_index_t HumanTuiPlayer<GameState_>::get_action(
    const GameState& state, const ActionMask& valid_actions)
{
  if (screen_clearing_enabled_) {
    util::clearscreen();
  }
  print_state(state);
  bool complain = false;
  int my_action = -1;
  while (true) {
    if (complain) {
      printf("Invalid input!\n");
    }
    complain = true;
    try {
      my_action = GameState::prompt_for_action();
      if (!valid_actions.test(my_action)) continue;
    } catch(...) {
      continue;
    }
    break;
  }

  return my_action;
}

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::print_state(const GameState& state) {
  state.dump(last_action_, &player_names_);
}

}  // namespace common
