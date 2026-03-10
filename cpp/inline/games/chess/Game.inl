#include "games/chess/Game.hpp"

#include "games/chess/MoveEncoder.hpp"
#include "core/BasicTypes.hpp"


namespace a0achess {

inline void Game::Rules::init_state(State& state) {
  state = GameState(chess::constants::STARTPOS);
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const InputFrame& frame) {
  return get_legal_moves(frame.to_state_unsafe());
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  chess::Movelist moves;
  chess::movegen::legalmoves(moves, state);

  Game::Types::ActionMask mask;
  for (const auto& move : moves) {
    core::action_t action = move_to_nn_idx(state, move);
    mask.set(action);
  }
  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.sideToMove() == chess::Color::WHITE ? kWhite : kBlack;
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  state.makeMove(nn_idx_to_move(state, action));
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t, core::action_t,
                                     GameResults::Tensor& outcome) {

  if (state.isHalfMoveDraw()) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (state.isInsufficientMaterial()) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (state.isRepetition(Game::Constants::kRepetitionDrawThreshold)) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  // TODO: this seems inefficient, to generate ALL moves, when we only need to know if there's at
  // least one legal move. We should consider adding this optimization into Disservin, and
  // submitting a PR to chess-library.
  chess::Movelist moves;
  chess::movegen::legalmoves(moves, state);

  if (moves.empty()) {
    if (state.inCheck()) {
      core::seat_index_t cp = get_current_player(state);
      outcome = core::WinLossDrawResults::win(1 - cp);
      return true;
    }
    // Stalemate.
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  return false;
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  return std::string(kMovesUCI[action]);
}

}  // namespace a0achess
