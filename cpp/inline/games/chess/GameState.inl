#include "games/chess/GameState.hpp"

#include "games/chess/MoveEncoder.hpp"

namespace a0achess {

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
}

inline core::action_t GameState::action_from_uci(const std::string& uci) const {
  chess::Move move = chess::uci::uciToMove(*this, uci);
  return move_to_nn_idx(*this, move);
}

}  // namespace a0achess
