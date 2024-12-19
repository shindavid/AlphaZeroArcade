#include <games/nim/Game.hpp>

namespace nim {

inline void Game::Rules::init_state(State& state) {
  state.set_stones(kStartingStones);
  state.set_player(0);
  state.set_player_ready(true);
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const StateHistory& history) {
  const State& state = history.current();
  Types::ActionMask mask;

  for (int i = 0; i < nim::kMaxStonesToTake; ++i) {
    mask[i] = i + 1 <= state.get_stones();
  }

  return mask;
}

inline void Game::Rules::apply_chance(StateHistory& history) {
  core::action_mode_t mode = get_action_mode(history.current());
  if (!is_chance_mode(mode)) {
    throw std::invalid_argument("Not in chance mode");
  }

  Types::PolicyTensor dist = get_chance_dist(history.current());
  core::action_t random_action = eigen_util::sample(dist);

  State& state = history.extend();
  int outcome_stones = std::max(state.get_stones() - random_action, 0);
  state.set_stones(outcome_stones);
  state.set_player_ready(true);
}

inline void Game::Rules::apply(StateHistory& history, core::action_t action) {
  if (action < 0 || action >= nim::kMaxStonesToTake) {
    throw std::invalid_argument("Invalid action: " + std::to_string(action));
  }

  State& state = history.extend();
  state.set_stones(state.get_stones() - (action + 1));
  state.set_player(1 - state.get_player());
  state.set_player_ready(false);
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
  if (state.get_stones() == 0) {
    outcome.setZero();
    outcome(last_player) = 1;
    return true;
  }
  return false;
}

inline Game::Types::PolicyTensor Game::Rules::get_chance_dist(const State& state) {
  core::action_mode_t mode = get_action_mode(state);
  if (!is_chance_mode(mode)) {
    throw std::invalid_argument("Not in chance mode");
  }

  Types::PolicyTensor dist;
  dist.setZero();
  for (int i = 0; i < nim::kMaxRandomStonesToTake + 1; ++i) {
    dist[i] = 1.0 / (nim::kMaxRandomStonesToTake + 1);
  }
  return dist;
}

}  // namespace nim