#include "games/chess/Game.hpp"
#include "core/BasicTypes.hpp"


namespace chess {

inline void Game::Rules::init_state(State& state) {
  state.reset();
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  Movelist moves = state.generate_legal_moves();

  Game::Types::ActionMask mask;
  for (const auto& move : moves) {
    core::action_t action = state.move_to_action(move);
    mask.set(action);
  }
  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.side_to_move() == Color::WHITE ? kWhite : kBlack;
}

inline void Game::Rules::apply(State& state, core::action_t action) { state.apply_action(action); }

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t, core::action_t,
                                     GameResults::Tensor& outcome) {
  Movelist legal_moves = state.generate_legal_moves();

  if (legal_moves.empty()) {
    if (state.in_check()) {
      core::seat_index_t cp = get_current_player(state);
      outcome = core::WinLossDrawResults::win(1 - cp);
      return true;
    }
    // Stalemate.
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (state.is_insufficient_material()) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (state.is_half_move_draw()) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (state.is_repetition(2)) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  return false;
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  return std::string(kMovesUCI[action]);
}

}  // namespace chess
