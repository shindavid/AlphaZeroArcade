#include "games/chess/Game.hpp"
#include "core/BasicTypes.hpp"
#include "lc0/chess/board.h"
#include "lc0/neural/encoder.h"

namespace chess {

inline void Game::Rules::init_state(State& state) {
  state.board = lczero::ChessBoard::kStartposBoard;
  state.recent_hashes.clear();
  state.zobrist_hash = state.board.Hash();
  state.history_hash = state.zobrist_hash;
  state.rule50_ply = 0;
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  // TODO: avoid std::vector
  const auto legal_moves = state.board.GenerateLegalMoves();
  Game::Types::ActionMask mask;
  for (const auto& move : legal_moves) {
    core::action_t action = static_cast<core::action_t>(lczero::MoveToNNIndex(move, 0));
    mask.set(action);
  }
  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.board.flipped() ? kBlack : kWhite;
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  auto move = lczero::MoveFromNNIndex(action, 0);
  bool reset_50_moves = state.board.ApplyMove(move);
  state.board.Mirror();

  // TODO: Optimization: only store the last board hash of the same player
  // (since only those are relevant for threefold repetition)
  // store the opponent's board hash for the next state
  state.recent_hashes.push_back(state.zobrist_hash);

  // TODO: Implement Zobrist hashing (lc0's hash is not Zobrist)
  state.zobrist_hash = state.board.Hash();

  if (reset_50_moves) {
    state.rule50_ply = 0;
    state.history_hash = state.zobrist_hash;
    state.recent_hashes.clear();
  } else {
    state.rule50_ply++;
    state.history_hash = lczero::HashCat({state.history_hash, state.zobrist_hash});
  }
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t, core::action_t,
                                     GameResults::Tensor& outcome) {
  const auto& board = state.board;
  auto legal_moves = board.GenerateLegalMoves();
  if (legal_moves.empty()) {
    if (board.IsUnderCheck()) {
      core::seat_index_t cp = get_current_player(state);
      outcome = core::WinLossDrawResults::win(1 - cp);
      return true;
    }
    // Stalemate.
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (!board.HasMatingMaterial()) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (state.rule50_ply >= 100) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  if (state.count_repetitions() >= 2) {
    outcome = core::WinLossDrawResults::draw();
    return true;
  }

  return false;
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  return lczero::MoveFromNNIndex(action, 0).ToString(false);
}

}  // namespace chess
