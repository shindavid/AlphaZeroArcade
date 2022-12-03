#include <connect4/C4HumanTuiPlayer.hpp>

#include <cstdlib>
#include <iostream>

#include <common/DerivedTypes.hpp>
#include <util/PrintUtil.hpp>
#include <util/ScreenUtil.hpp>

namespace c4 {

inline void HumanTuiPlayer::start_game(const player_array_t& players, common::player_index_t seat_assignment) {
  for (int p = 0; p < int(player_names_.size()); ++p) {
    player_names_[p] = players[p]->get_name();
  }
  my_index_ = seat_assignment;
  util::set_xprintf_target(buf_);
}

inline void HumanTuiPlayer::receive_state_change(
    common::player_index_t, const GameState& state, common::action_index_t action, const Result& result)
{
  last_action_ = action;
  if (common::is_terminal_result(result)) {
    xprintf_switch(state);
  }
}

inline common::action_index_t HumanTuiPlayer::get_action(const GameState& state, const ActionMask& valid_actions) {
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
    std::cout << "Enter move [1-7]: ";
    std::cout.flush();
    std::string input;
    std::getline(std::cin, input);
    try {
      my_action = std::stoi(input) - 1;
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

inline void HumanTuiPlayer::xprintf_switch(const GameState& state) {
  util::clear_xprintf_target();
  print_state(state);
  std::cout << buf_.str();
  buf_.str("");
  buf_.clear();
  std::cout.flush();
}

inline void HumanTuiPlayer::print_state(const GameState& state) {
  util::clearscreen();
  state.xprintf_dump(player_names_, last_action_);
}

}  // namespace c4
