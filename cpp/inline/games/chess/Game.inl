#include "games/chess/Game.hpp"

#include "core/BasicTypes.hpp"

namespace a0achess {

inline void Game::Rules::init_state(State& state) { state.init(); }

inline Game::Rules::Result Game::Rules::analyze(const InputFrame& frame) {
  return analyze(frame.to_state_unsafe());
}

inline core::game_phase_t Game::Rules::get_game_phase(const State& state) {
  return state.sideToMove() == chess::Color::WHITE ? kWhiteToMove : kBlackToMove;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.sideToMove() == chess::Color::WHITE ? kWhite : kBlack;
}

inline Game::Rules::Result Game::Rules::analyze(const State& state) {
  if (state.isHalfMoveDraw()) return PlayerResult::make_draw<Constants::kNumPlayers>();
  if (state.isInsufficientMaterial()) return PlayerResult::make_draw<Constants::kNumPlayers>();
  if (state.isRepetition(Game::Constants::kRepetitionDrawThreshold))
    return PlayerResult::make_draw<Constants::kNumPlayers>();

  MoveSet moves;
  chess::movegen::legalmoves(moves, state);

  if (moves.empty()) {
    if (state.inCheck()) {
      auto winner = 1 - get_current_player(state);
      return PlayerResult::make_win<Constants::kNumPlayers>(winner);
    }
    // Stalemate.
    return PlayerResult::make_draw<Constants::kNumPlayers>();
  }

  return moves;
}

}  // namespace a0achess
