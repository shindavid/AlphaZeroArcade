#include <games/nim/Game.hpp>

namespace nim {

inline void Game::Rules::init_state(State& state) {
  state.stones_left = kStartingStones;
  state.next_player = 0;
}

inline size_t Game::State::hash() const {
  auto tuple = std::make_tuple(stones_left, next_player, chance_active);
  std::hash<decltype(tuple)> hasher;
  return hasher(tuple);
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const StateHistory& history) {
  const State& state = history.current();
  Types::ActionMask mask;
  bool is_chance = is_chance_mode(get_action_mode(history.current()));
  int max_num_action = is_chance ? nim::kChanceDistributionSize : nim::kMaxStonesToTake;

  for (int i = 0; i < std::min(max_num_action, state.stones_left); ++i) {
    mask[i] = true;
  }

  return mask;
}

inline void Game::Rules::apply(StateHistory& history, core::action_t action) {
  bool is_chance = is_chance_mode(get_action_mode(history.current()));
  State& state = history.extend();

  if (is_chance) {
    int outcome_stones = state.stones_left - action;
    state.stones_left = outcome_stones;
    state.next_player = 1 - state.next_player;
    state.chance_active = false;
  } else {
    if (action < 0 || action >= nim::kMaxStonesToTake) {
      throw std::invalid_argument("Invalid action: " + std::to_string(action));
    }
    state.stones_left = state.stones_left- (action + 1);
    state.chance_active = true;
  }
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
  if (state.stones_left == 0) {
    outcome.setZero();
    outcome(last_player) = 1;
    return true;
  }
  return false;
}

inline Game::Types::ChanceDistribution Game::Rules::get_chance_distribution(const State& state) {
  if (!is_chance_mode(get_action_mode(state))) {
    throw std::invalid_argument("Not in chance mode");
  }

  Types::ChanceDistribution dist;
  dist.setZero();
  for (int i = 0; i < nim::kChanceDistributionSize; ++i) {
    dist(i) = nim::kChanceEventProbs[i];
  }
  return dist;
}

template <typename Iter>
inline Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(Iter start, Iter cur) {
  Tensor tensor;
  tensor.setZero();
  Iter state = cur;

  constexpr int bit_width = std::bit_width(kStartingStones);
  for (int i = 0; i < bit_width; ++i) {
    tensor(i) = (state->stones_left & (1 << i)) ? 1 : 0;
  }
  tensor(bit_width) = state->next_player;
  tensor(bit_width + 1) = state->chance_active;
  return tensor;
}
}  // namespace nim
