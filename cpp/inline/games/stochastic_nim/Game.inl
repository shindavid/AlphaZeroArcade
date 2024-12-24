#include <games/stochastic_nim/Game.hpp>

namespace stochastic_nim {

inline void Game::Rules::init_state(State& state) {
  state.stones_left = kStartingStones;
  state.current_player = 0;
}

inline size_t Game::State::hash() const {
  auto tuple = std::make_tuple(stones_left, current_player, current_mode);
  std::hash<decltype(tuple)> hasher;
  return hasher(tuple);
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const StateHistory& history) {
  const State& state = history.current();
  Types::ActionMask mask;
  bool is_chance = is_chance_mode(state.current_mode);
  if (is_chance) {
    for (int i = 0; i < std::min(stochastic_nim::kChanceDistributionSize, state.stones_left + 1); ++i) {
      mask[i] = true;
    }
  } else {
    for (int i = 0; i < std::min(stochastic_nim::kMaxStonesToTake, state.stones_left); ++i) {
      mask[i] = true;
    }
  }

  return mask;
}

// current_player only switches AFTER a chance action
inline void Game::Rules::apply(StateHistory& history, core::action_t action) {
  bool is_chance = is_chance_mode(history.current().current_mode);
  State& state = history.extend();

  if (is_chance) {
    int outcome_stones = state.stones_left - action;
    state.stones_left = outcome_stones;
    state.current_player = 1 - state.current_player;
    state.current_mode = stochastic_nim::kPlayerMode;
  } else {
    if (action < 0 || action >= stochastic_nim::kMaxStonesToTake) {
      throw std::invalid_argument("Invalid action: " + std::to_string(action));
    }
    state.stones_left = state.stones_left - (action + 1);
    state.current_mode = stochastic_nim::kChanceMode;
  }
}

// if the game ends after a chance action, the player who made the last move wins
inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
  if (state.stones_left == 0) {
    outcome.setZero();
    outcome(last_player) = 1;
    return true;
  }
  return false;
}

/**
 * Assign the chance distribution mass to each legal move. If the sum of the probabilities is less
 * than 1, move the remaining probability mass to the last legal move.
 */
inline Game::Types::ChanceDistribution Game::Rules::get_chance_distribution(const State& state) {
  if (!is_chance_mode(get_action_mode(state))) {
    throw std::invalid_argument("Not in chance mode");
  }
  int num_legal_moves = std::min(stochastic_nim::kChanceDistributionSize, state.stones_left + 1);
  Types::ChanceDistribution dist;
  dist.setZero();

  float cumulative_prob = 0;
  for (int i = 0; i < num_legal_moves; ++i) {
    dist(i) = stochastic_nim::kChanceEventProbs[i];
    cumulative_prob += dist(i);
  }
  dist(num_legal_moves - 1) += 1 - cumulative_prob;
  return dist;
}

template <typename Iter>
inline Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(Iter start, Iter cur) {
  Tensor tensor;
  tensor.setZero();
  Iter state = cur;

  for (int i = 0; i < stochastic_nim::kStartingStonesBitWidth; ++i) {
    tensor(i) = (state->stones_left & (1 << i)) ? 1 : 0;
  }
  tensor(stochastic_nim::kStartingStonesBitWidth) = state->current_player;
  tensor(stochastic_nim::kStartingStonesBitWidth + 1) = state->current_mode;
  return tensor;
}
}  // namespace stochastic_nim

