#include <common/HumanTuiPlayer.hpp>

#include <cstdlib>
#include <iostream>

#include <common/DerivedTypes.hpp>
#include <util/ScreenUtil.hpp>

namespace common {

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::receive_state_change(
    common::player_index_t p, const GameState& state, common::action_index_t action)
{
  last_action_ = action;
}

template<GameStateConcept GameState_>
inline common::action_index_t HumanTuiPlayer<GameState_>::get_action(
    const GameState& state, const ActionMask& valid_actions)
{
  util::ScreenClearer::clear_once();
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

  util::ScreenClearer::reset();
  return my_action;
}

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::end_game(const GameState& state, const GameOutcome&) {
  util::ScreenClearer::clear_once();
  HumanTuiPlayer::print_state(state);
}

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::print_state(const GameState& state) {
  state.dump(last_action_, &this->get_player_names());
}

}  // namespace common
