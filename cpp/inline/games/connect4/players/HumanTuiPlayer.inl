#include "games/connect4/players/HumanTuiPlayer.hpp"

#include "util/Asserts.hpp"

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

inline bool HumanTuiPlayer::start_game() {
  if (move_history_) move_history_->reset();
  return base_t::start_game();
}

inline void HumanTuiPlayer::receive_state_change(const StateChangeUpdate& update) {
  if (move_history_) {
    // TODO: to support undos in cheat-mode, we need to do a specialized update to move_history_
    // here. Probably simplest to just rewrite it from scratch by crawling the update's state tree
    // iterator.
    RELEASE_ASSERT(!update.is_jump(), "undo not yet supported in cheat mode");
    move_history_->append(*update.move());
  }
  base_t::receive_state_change(update);
}

inline HumanTuiPlayer::ActionResponse HumanTuiPlayer::prompt_for_action(
  const ActionRequest& request) {
  bool undo_allowed = request.undo_allowed;

  if (undo_allowed) {
    std::cout << "Enter move [1-7] or U to undo: ";
  } else {
    std::cout << "Enter move [1-7]: ";
  }

  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);

  if (input == "U" || input == "u") {
    if (undo_allowed) {
      return ActionResponse::undo();
    } else {
      return ActionResponse::invalid();
    }
  }

  try {
    return Move(std::stoi(input) - 1);
  } catch (std::invalid_argument& e) {
    return ActionResponse::invalid();
  } catch (std::out_of_range& e) {
    return ActionResponse::invalid();
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

  Game::IO::print_state(std::cout, state, last_move_set_ ? &last_move_ : nullptr,
                        &this->get_player_names());
}

}  // namespace c4
