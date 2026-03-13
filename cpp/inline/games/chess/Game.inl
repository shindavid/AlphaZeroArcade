#include "games/chess/Game.hpp"

#include "core/BasicTypes.hpp"
#include "games/chess/MoveEncoder.hpp"

namespace a0achess {

inline void Game::Rules::init_state(State& state) { state = GameState(chess::constants::STARTPOS); }

inline Game::Rules::Result Game::Rules::analyze(const InputFrame& frame,
                                                const core::MoveInfo& last_move_info) {
  return analyze(frame.to_state_unsafe(), last_move_info);
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.sideToMove() == chess::Color::WHITE ? kWhite : kBlack;
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  state.makeMove(nn_idx_to_move(state, action));
}

inline Game::Rules::Result Game::Rules::analyze(const State& state,
                                                const core::MoveInfo& last_move_info) {
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

  if (moves.empty()) {
    if (state.inCheck()) {
      core::seat_index_t cp = get_current_player(state);
      return Result::make_terminal(GameResults::win(1 - cp));
    }
    // Stalemate.
    return Result::make_terminal(GameResults::draw());
  }

  Game::Types::ActionMask mask;
  for (const auto& move : moves) {
    core::action_t action = move_to_nn_idx(state, move);
    mask.set(action);
  }
  return Result::make_nonterminal(mask);
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  return std::string(kMovesUCI[action]);
}

}  // namespace a0achess
