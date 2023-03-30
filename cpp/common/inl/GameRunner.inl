#include <common/GameRunner.hpp>

#include <common/DerivedTypes.hpp>
#include <util/CppUtil.hpp>
#include <util/Exception.hpp>
#include <util/Random.hpp>

namespace common {

template<GameStateConcept GameState>
typename GameRunner<GameState>::GameOutcome GameRunner<GameState>::run(PlayerOrder order) {
  game_id_t game_id = util::get_unique_id();
  if (order == kRandomPlayerSeats) {
    util::Random::shuffle(players_.begin(), players_.end());
  }
  player_name_array_t player_names;
  for (size_t p = 0; p < players_.size(); ++p) {
    player_names[p] = players_[p]->get_name();
  }
  for (size_t p = 0; p < players_.size(); ++p) {
    players_[p]->init_game(game_id, player_names, p);
    players_[p]->start_game();
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
    auto outcome = state.apply_move(action);
    for (auto player2 : players_) {
      player2->receive_state_change(p, state, action, outcome);
    }
    if (is_terminal_outcome(outcome)) {
      return outcome;
    }
  }
  throw std::runtime_error("should not get here");
}

}  // namespace common
