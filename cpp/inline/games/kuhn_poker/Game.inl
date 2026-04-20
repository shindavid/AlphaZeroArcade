#include "games/kuhn_poker/Game.hpp"

namespace kuhn_poker {

namespace {

inline Game::Rules::Result showdown(const GameState& state) {
  using GameOutcome = Game::Types::GameOutcome;
  // Pot size: 1 (ante only) if check-check, 2 (ante + bet) if bet-call
  bool has_bet = false;
  for (int i = 0; i < state.num_actions; ++i) {
    if (state.actions[i] == kBet) {
      has_bet = true;
      break;
    }
  }
  float pot = has_bet ? 2.0f : 1.0f;
  GameOutcome outcome;
  if (state.cards[0] > state.cards[1]) {
    outcome[0].score = pot;
    outcome[1].score = -pot;
  } else {
    outcome[0].score = -pot;
    outcome[1].score = pot;
  }
  return outcome;
}

}  // anonymous namespace

inline void Game::Rules::init_state(State& state) {
  state = State{};  // deal phase, cards undealt
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.current_player;
}

inline bool Game::Rules::is_chance_state(const State& state) { return state.phase == kDealPhase; }

inline ChanceDistribution Game::Rules::get_chance_distribution(const State& state) {
  return ChanceDistribution(state);
}

inline void Game::Rules::apply(State& state, const Move& move) {
  if (state.phase == kDealPhase) {
    int deal_ix = move.index();
    state.cards[0] = kDealTable[deal_ix][0];
    state.cards[1] = kDealTable[deal_ix][1];
    state.phase = kBettingPhase;
    state.current_player = 0;
  } else {
    int action = move.index();
    state.actions[state.num_actions] = action;
    state.num_actions++;

    if (action == kCheck || action == kBet) {
      state.current_player = 1 - state.current_player;
    }
    // Fold and Call do not switch player (game ends after these)
  }
}

inline Game::Rules::Result Game::Rules::analyze(const State& state) {
  if (state.phase == kDealPhase) {
    MoveSet moves;
    for (int i = 0; i < kNumDeals; ++i) {
      moves.add(Move(i, kDealPhase));
    }
    return moves;
  }

  int n = state.num_actions;

  // Check terminal conditions
  if (n >= 1) {
    int last = state.actions[n - 1];

    if (last == kFold) {
      // The folder is current_player (apply doesn't switch on fold)
      core::seat_index_t folder = state.current_player;
      core::seat_index_t winner = 1 - folder;
      GameOutcome outcome;
      outcome[winner].score = 1.0f;   // win the ante
      outcome[folder].score = -1.0f;  // lose the ante
      return outcome;
    }

    if (last == kCall) {
      return showdown(state);
    }
  }

  if (n >= 2) {
    int last = state.actions[n - 1];
    int prev = state.actions[n - 2];
    if (last == kCheck && prev == kCheck) {
      return showdown(state);
    }
  }

  // Game is not over — determine legal moves
  MoveSet legal_moves;
  if (n == 0) {
    // Player 0's first action: can check or bet
    legal_moves.add(Move(kCheck, kBettingPhase));
    legal_moves.add(Move(kBet, kBettingPhase));
  } else {
    int last_action = state.actions[n - 1];
    if (last_action == kCheck) {
      legal_moves.add(Move(kCheck, kBettingPhase));
      legal_moves.add(Move(kBet, kBettingPhase));
    } else if (last_action == kBet) {
      legal_moves.add(Move(kFold, kBettingPhase));
      legal_moves.add(Move(kCall, kBettingPhase));
    }
  }

  return legal_moves;
}

inline Game::InfoSet Game::Rules::state_to_info_set(const State& state, core::seat_index_t seat) {
  InfoSet info;
  info.my_card = state.cards[seat];
  info.current_player = state.current_player;
  info.phase = state.phase;
  info.num_actions = state.num_actions;
  for (int i = 0; i < state.num_actions; ++i) {
    info.actions[i] = state.actions[i];
  }
  return info;
}

inline std::string Game::IO::player_to_str(core::seat_index_t player) {
  return std::format("P{}", player);
}

inline void Game::IO::print_state(std::ostream& os, const State& state, const Move* last_move,
                                  const Types::player_name_array_t* player_names) {
  os << compact_state_repr(state) << std::endl;
}

inline std::string Game::IO::compact_state_repr(const State& state) {
  std::ostringstream ss;
  if (state.phase == kDealPhase) {
    ss << "deal";
  } else {
    ss << kCardNames[state.cards[0]] << kCardNames[state.cards[1]];
    for (int i = 0; i < state.num_actions; ++i) {
      ss << "-" << kActionNames[state.actions[i]];
    }
  }
  return ss.str();
}

}  // namespace kuhn_poker
