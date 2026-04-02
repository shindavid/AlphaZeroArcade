#include "games/stochastic_nim/Game.hpp"

namespace stochastic_nim {

inline void Game::Rules::init_state(State& state) {
  state.stones_left = kStartingStones;
  state.current_player = 0;
  state.last_player = -1;
  state.current_phase = kPlayerPhase;
}

inline MoveList Game::Rules::get_legal_moves(const State& state) {
  MoveList moves;
  bool is_chance = is_chance_phase(state.current_phase);
  if (is_chance) {
    for (int i = 0; i < std::min(stochastic_nim::kChanceDistributionSize, state.stones_left + 1);
         ++i) {
      moves.add(Move(i, stochastic_nim::kChancePhase));
    }
  } else {
    for (int i = 0; i < std::min(stochastic_nim::kMaxStonesToTake, state.stones_left); ++i) {
      moves.add(Move(i, stochastic_nim::kPlayerPhase));
    }
  }
  return moves;
}

inline core::game_phase_t Game::Rules::get_game_phase(const State& state) {
  return state.current_phase;
}
inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.current_player;
}

// current_player only switches AFTER a chance action
inline void Game::Rules::apply(State& state, const Move& move) {
  bool is_chance = is_chance_phase(state.current_phase);

  if (is_chance) {
    int outcome_stones = state.stones_left - move.index();
    state.stones_left = outcome_stones;
    state.current_player = 1 - state.current_player;
    state.current_phase = stochastic_nim::kPlayerPhase;
  } else {
    if (move.index() < 0 || move.index() >= stochastic_nim::kMaxStonesToTake) {
      throw std::invalid_argument("Invalid action: " + std::to_string(move.index()));
    }
    state.stones_left = state.stones_left - (move.index() + 1);
    state.current_phase = stochastic_nim::kChancePhase;
    state.last_player = state.current_player;
  }
}

inline constexpr bool Game::Rules::is_chance_phase(core::game_phase_t phase) {
  return phase == stochastic_nim::kChancePhase;
}

/*
 * Assign the chance distribution mass to each legal move. If the sum of the probabilities is less
 * than 1, move the remaining probability mass to the last legal move.
 */
inline ChanceDistribution Game::Rules::get_chance_distribution(const State& state) {
  if (!is_chance_phase(get_game_phase(state))) {
    throw std::invalid_argument("Not in chance phase");
  }
  return ChanceDistribution(state);
}

inline void Game::IO::print_state(std::ostream& ss, const State& state, const Move& last_move,
                                  const Types::player_name_array_t* player_names) {
  ss << compact_state_repr(state) << std::endl;
}

inline std::string Game::IO::compact_state_repr(const State& state) {
  std::ostringstream ss;
  ss << "p" << state.current_player;
  if (state.current_phase == stochastic_nim::kChancePhase) {
    ss << "*";
  }
  ss << "@" << state.stones_left;
  return ss.str();
}

// if the game ends after a chance action, the player who made the last move wins
inline Game::Rules::Result Game::Rules::analyze(const State& state) {
  if (state.stones_left == 0) {
    GameResults::Tensor outcome;
    outcome.setZero();
    outcome(state.last_player) = 1;
    return Result::make_terminal(outcome);
  }
  return Result::make_nonterminal(get_legal_moves(state));
}

}  // namespace stochastic_nim
