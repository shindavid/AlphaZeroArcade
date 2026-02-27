#include "chess-library/src/uci.hpp"
#include "games/chess/GameState.hpp"

namespace chess {

inline Movelist GameState::generate_legal_moves() const {
  Movelist moves;
  movegen::legalmoves(moves, board_);
  return moves;
}

inline void GameState::apply_action(core::action_t action) {
  Move move = chess::nn_idx_to_move(board_, action);
  board_.makeMove(move);
}

inline core::action_t GameState::action_from_uci(const std::string& uci) const {
  Move move = chess::uci::uciToMove(board_, uci);
  return move_to_nn_idx(board_, move);
}

}  // namespace chess
