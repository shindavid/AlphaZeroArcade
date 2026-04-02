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
  if (state.isHalfMoveDraw()) {
    return Result::make_terminal(GameResults::draw());
  }

  if (state.isInsufficientMaterial()) {
    return Result::make_terminal(GameResults::draw());
  }

  if (state.isRepetition(Game::Constants::kRepetitionDrawThreshold)) {
    return Result::make_terminal(GameResults::draw());
  }

  chess::Movelist moves;
  chess::movegen::legalmoves(moves, state);

  MoveList valid_moves;
  auto phase = get_game_phase(state);
  for (const chess::Move& move : moves) {
    valid_moves.add(Move(move, phase));
  }

  if (moves.empty()) {
    if (state.inCheck()) {
      core::seat_index_t cp = get_current_player(state);
      return Result::make_terminal(GameResults::win(1 - cp));
    }
    // Stalemate.
    return Result::make_terminal(GameResults::draw());
  }

  return Result::make_nonterminal(valid_moves);
}

}  // namespace a0achess
