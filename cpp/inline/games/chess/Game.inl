#include "games/chess/Game.hpp"
#include "lc0/chess/board.h"
#include "lc0/neural/encoder.h"

namespace chess {

inline void Game::Rules::init_state(State& state) {
  state.board = lczero::ChessBoard::kStartposBoard;
  state.recent_hashes.clear();
  state.zobrist_hash = 0;
  state.history_hash = 0;
  state.rule50_ply = 0;
  state.seat = 0;
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  const auto legal_moves = state.board.GenerateLegalMoves();
  Game::Types::ActionMask mask;
  for (const auto& move : legal_moves) {
    core::action_t action = static_cast<core::action_t>(lczero::MoveToNNIndex(move, 0));
    mask.set(action);
  }
  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.seat;
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  auto move = lczero::MoveFromNNIndex(action, 0);
  bool reset_50_moves = state.board.ApplyMove(move);
  if (reset_50_moves) {
    state.rule50_ply = 0;
  } else {
    state.rule50_ply++;
  }
  state.zobrist_hash = state.board.Hash();
  state.history_hash = 0; //boost::hash_combine(state.history_hash, state.zobrist_hash);
  state.recent_hashes.push_back(state.zobrist_hash);

  state.seat = 1 - state.seat;
  state.board.Mirror();
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
  throw std::runtime_error("Not implemented");
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  return lczero::MoveFromNNIndex(action, 0).ToString(false);
}

}  // namespace chess
