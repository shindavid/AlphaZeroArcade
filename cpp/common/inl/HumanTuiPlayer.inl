#include <common/HumanTuiPlayer.hpp>

#include <cstdlib>
#include <iostream>

#include <common/DerivedTypes.hpp>
#include <util/PrintUtil.hpp>
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
  util::set_xprintf_target(buf_);
}

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::receive_state_change(
    common::player_index_t, const GameState& state, common::action_index_t action, const GameOutcome& outcome)
{
  last_action_ = action;
  if (common::is_terminal_outcome(outcome)) {
    xprintf_switch(state);
  }
}

template<GameStateConcept GameState_>
inline common::action_index_t HumanTuiPlayer<GameState_>::get_action(
    const GameState& state, const ActionMask& valid_actions)
{
  xprintf_switch(state);

  bool complain = false;
  int my_action = -1;
  while (true) {
    if (complain) {
      print_state(state);
      printf("%s", buf_.str().c_str());
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

  buf_.clear();
  util::set_xprintf_target(buf_);
  return my_action;
}

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::xprintf_switch(const GameState& state) {
  util::clear_xprintf_target();
  print_state(state);
  std::cout << buf_.str();
  buf_.str("");
  buf_.clear();
  std::cout.flush();
}

template<GameStateConcept GameState_>
inline void HumanTuiPlayer<GameState_>::print_state(const GameState& state) {
  util::clearscreen();
  state.xprintf_dump(player_names_, last_action_);
}

}  // namespace common
