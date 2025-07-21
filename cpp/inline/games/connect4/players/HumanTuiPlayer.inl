#include "games/connect4/players/HumanTuiPlayer.hpp"

#include <iostream>
#include <string>

namespace c4 {

inline HumanTuiPlayer::HumanTuiPlayer(bool cheat_mode) {
  if (cheat_mode) {
    oracle_ = new PerfectOracle();
    move_history_ = new PerfectOracle::MoveHistory();
  }
}

inline HumanTuiPlayer::~HumanTuiPlayer() {
  delete oracle_;
  delete move_history_;
}

inline void HumanTuiPlayer::start_game() {
  if (move_history_) move_history_->reset();
  base_t::start_game();
}

inline void HumanTuiPlayer::receive_state_change(core::seat_index_t seat, const State& state,
                                                 core::action_t action) {
  if (move_history_) move_history_->append(action);
  base_t::receive_state_change(seat, state, action);
}

inline core::action_t HumanTuiPlayer::prompt_for_action(const State&, const ActionMask&) {
  std::cout << "Enter move [1-7]: ";
  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);
  try {
    return std::stoi(input) - 1;
  } catch (std::invalid_argument& e) {
    return -1;
  } catch (std::out_of_range& e) {
    return -1;
  }
}

inline void HumanTuiPlayer::print_state(const State& state, bool terminal) {
  if (oracle_) {
    if (terminal) {
      std::cout << std::endl;  // blank link to retain alignment
    } else {
      auto result = oracle_->query(*move_history_);
      printf("%s\n", result.get_overlay().c_str());
    }
  }

  Game::IO::print_state(std::cout, state, last_action_, &this->get_player_names());
}

}  // namespace c4
