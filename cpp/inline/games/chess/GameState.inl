#include "games/chess/GameState.hpp"

#include "games/chess/Constants.hpp"
#include "util/Exceptions.hpp"

#include <chess-library/include/chess.hpp>

namespace a0achess {

inline void GameState::init() {
  *this = GameState(chess::constants::STARTPOS);
  this->history_hash_ = this->key_;
}

inline void GameState::makeMove(const chess::Move& move) {
  chess::Board::makeMove(move);

  // halfmove clock resets to 0 on pawn moves and captures (irreversible moves)
  if (halfMoveClock() == 0) {
    history_hash_ = hash();
  } else {
    history_hash_ = history_hash_ * kHistoryHashRollConstant + hash();
  }
}

// TODO: I recently added a change to Disservin to add a backtrackTo() method, and the PR was
// merged into master. After we upgrade our git subtree of Disservin, we should change this method
// to call backtrackTo() to update all the chess::Board members, followed by simply setting the
// history_hash_.
inline void GameState::backtrack_to(const GameState& prev_state) {
  this->backtrackTo(prev_state);
  this->history_hash_ = prev_state.history_hash_;
}

inline void GameState::dump_recent_hashes(std::ostringstream& ss, int n_prev_states_to_dump) const {
  const auto size = static_cast<int>(prev_states_.size());

  for (int i = std::max(0, size - n_prev_states_to_dump); i < size; ++i) {
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

}  // namespace a0achess
