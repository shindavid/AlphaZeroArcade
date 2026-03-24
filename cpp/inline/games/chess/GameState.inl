#include "games/chess/GameState.hpp"

#include "games/chess/MoveEncoder.hpp"

namespace a0achess {

inline void GameState::init() {
  *this = GameState(chess::constants::STARTPOS);
  this->history_hash_ = this->key_;
}

inline void GameState::backtrack_to(const GameState& prev_state) {
  int n = prev_state.prev_states_.size();
  this->prev_states_.erase(this->prev_states_.begin() + n, this->prev_states_.end());

  this->pieces_bb_ = prev_state.pieces_bb_;
  this->occ_bb_ = prev_state.occ_bb_;
  this->board_ = prev_state.board_;
  this->key_ = prev_state.key_;
  this->cr_ = prev_state.cr_;
  this->plies_ = prev_state.plies_;
  this->stm_ = prev_state.stm_;
  this->ep_sq_ = prev_state.ep_sq_;
  this->hfm_ = prev_state.hfm_;
  this->chess960_ = prev_state.chess960_;
  this->castling_path = prev_state.castling_path;
  this->history_hash_ = prev_state.history_hash_;
}

inline void GameState::apply_move(core::action_t action) {
  chess::Move move = nn_idx_to_move(*this, action);
  makeMove(move);

  // halfmove clock resets to 0 on pawn moves and captures (irreversible moves)
  if (halfMoveClock() == 0) {
    history_hash_ = hash();
  } else {
    history_hash_ = history_hash_ * 0x9e3779b97f4a7c15UL + hash();
  }
}

inline core::action_t GameState::action_from_uci(const std::string& uci) const {
  chess::Move move = chess::uci::uciToMove(*this, uci);
  return move_to_nn_idx(*this, move);
}

}  // namespace a0achess
