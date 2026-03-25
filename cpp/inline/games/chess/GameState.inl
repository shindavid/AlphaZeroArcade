#include "games/chess/GameState.hpp"

#include "games/chess/MoveEncoder.hpp"

namespace a0achess {

inline void GameState::init() {
  *this = GameState(chess::constants::STARTPOS);
  this->history_hash_ = this->key_;
}

inline void GameState::backtrack_to(const GameState& prev_state) {
  RELEASE_ASSERT(prev_state.prev_states_.size() <= prev_states_.size());
  int n = prev_state.prev_states_.size();

  if (IS_DEFINED(DEBUG_BUILD)) {
    int n_prev_states_check = 5;

    // check that for the last n_prev_states_check states, the hashes match up
    int start = std::max(0, n - n_prev_states_check);
    for (int i = start; i < n; ++i) {
      if (prev_states_[i].hash != prev_state.prev_states_[i].hash) {
        throw util::Exception("prev_states_[{}] hash mismatch {} != {}", i, prev_states_[i].hash,
                              prev_state.prev_states_[i].hash);
      }
    }
  }

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
    history_hash_ = history_hash_ * kHistoryHashRollConstant + hash();
  }
}

inline void GameState::dump_recent_hashes(std::ostringstream& ss, int n_prev_states_to_dump) const {
  const auto size = static_cast<int>(prev_states_.size());

  for (int i = std::max(0, size-n_prev_states_to_dump); i < size; ++i) {
    ss << std::format("{:4d} {}", i, prev_states_[i].hash) << std::endl;
  }
  ss << std::format("{:4d} {}", size, key_) << std::endl;
}

inline void GameState::validate_history_hash() const {
  uint64_t h = 0;
  int c = halfMoveClock();

  int n = prev_states_.size();
  for (int i = n - c; i < n; ++i) {
    h = h * kHistoryHashRollConstant + prev_states_[i].hash;
  }
  h = h * kHistoryHashRollConstant + key_;

  if (h != history_hash_) {
    throw util::Exception("History hash mismatch {} != {}", h, history_hash_);
  }
}

inline core::action_t GameState::action_from_uci(const std::string& uci) const {
  chess::Move move = chess::uci::uciToMove(*this, uci);
  return move_to_nn_idx(*this, move);
}

}  // namespace a0achess
