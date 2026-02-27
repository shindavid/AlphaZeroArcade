#include "chess-library/src/uci.hpp"
#include "games/chess/GameState.hpp"

namespace chess {

inline void GameState::reset() {
  board_ = Board(chess::constants::STARTPOS);
  past_hashes_.clear();
}

inline Movelist GameState::generate_legal_moves() const {
  Movelist moves;
  movegen::legalmoves(moves, board_);
  return moves;
}

inline void GameState::apply_action(core::action_t action) {
  past_hashes_.push_back(board_.hash());
  Move move = chess::nn_idx_to_move(board_, action);
  board_.makeMove(move);
}

inline core::action_t GameState::action_from_uci(const std::string& uci) const {
  Move move = chess::uci::uciToMove(board_, uci);
  return move_to_nn_idx(board_, move);
}

inline bool GameState::is_repetition(int repetitions) const {
  zobrist_hash_t current_hash = board_.hash();
  std::uint8_t c = 0;
  int hfm = half_move_clock();

  // We start the loop from the back and go forward in moves, at most to the
  // last move which reset the half-move counter because repetitions cant
  // be across half-moves.
  const auto size = static_cast<int>(past_hashes_.size());

  for (int i = size - 2; i >= 0 && i >= size - hfm - 1; i -= 2) {
    if (*(past_hashes_.begin() + i) == current_hash) c++;
    if (c == repetitions) return true;
  }

  return false;
}

}  // namespace chess
