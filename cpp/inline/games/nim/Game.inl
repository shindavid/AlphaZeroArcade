#include <games/nim/Game.hpp>

namespace nim {

inline void Game::Rules::init_state(State& state) {
  state.set_stones(kStartingStones);
  state.set_player(0);
  state.set_player_ready(true);
}

inline size_t Game::State::hash() const {
  auto tuple = std::make_tuple(stones_left, current_player, player_ready);
  std::hash<decltype(tuple)> hasher;
  return hasher(tuple);
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const StateHistory& history) {
  const State& state = history.current();
  Types::ActionMask mask;
  bool is_chance = Rules::has_known_dist(history.current());

  if (is_chance) {
    for (int i = 0; i < kMaxRandomStonesToTake + 1; ++i) {
      mask[i] = true;
    }
  } else {
    for (int i = 0; i < nim::kMaxStonesToTake; ++i) {
      mask[i] = i + 1 <= state.get_stones();
    }
  }
  return mask;
}

inline void Game::Rules::apply(StateHistory& history, core::action_t action) {
  bool is_chance = has_known_dist(history.current());
  State& state = history.extend();

  if (is_chance) {
    int outcome_stones = std::max(state.get_stones() - action, 0);
    state.set_stones(outcome_stones);
    state.set_player_ready(true);
  } else {
    if (action < 0 || action >= nim::kMaxStonesToTake) {
      throw std::invalid_argument("Invalid action: " + std::to_string(action));
    }
    state.set_stones(state.get_stones() - (action + 1));
    state.set_player(1 - state.get_player());
    if (kMaxRandomStonesToTake > 0) {
      state.set_player_ready(false);
    }
  }
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

inline Game::Types::PolicyTensor Game::Rules::get_known_dist(const State& state) {
  if (!has_known_dist(state)) {
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