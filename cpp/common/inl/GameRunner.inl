#include <common/GameRunner.hpp>

#include <common/DerivedTypes.hpp>
#include <util/Exception.hpp>

namespace common {

template<GameStateConcept GameState>
typename GameRunner<GameState>::Result GameRunner<GameState>::run() {
  for (size_t p = 0; p < players_.size(); ++p) {
    players_[p]->start_game(players_, p);
  }

  GameState state;
  while (true) {
    player_index_t p = state.get_current_player();
    Player* player = players_[p];
    auto valid_actions = state.get_valid_actions();
    action_index_t action = player->get_action(state, valid_actions);
    if (!valid_actions[action]) {
      throw util::Exception("Player %d (%s) attempted an illegal action (%d)", p, player->get_name().c_str(), action);
    }
    auto result = state.apply_move(action);
    for (auto player2 : players_) {
      player2->receive_state_change(p, state, action, result);
    }
    if (is_terminal_result(result)) {
      return result;
    }
  }
  throw std::runtime_error("should not get here");
}

}  // namespace common
