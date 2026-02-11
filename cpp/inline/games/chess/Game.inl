#include "games/chess/Game.hpp"
#include "core/BasicTypes.hpp"
#include "lc0/chess/board.h"
#include "lc0/neural/encoder.h"

namespace chess {

inline void Game::Rules::init_state(State& state) {
  state.board = lczero::ChessBoard::kStartposBoard;
  state.recent_hashes.clear();
  state.zobrist_hash = 0;
  state.history_hash = 0;
  state.rule50_ply = 0;
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
  return state.board.flipped() ? 1 : 0;
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  auto move = lczero::MoveFromNNIndex(action, 0);
  bool reset_50_moves = state.board.ApplyMove(move);
  state.board.Mirror();

  if (reset_50_moves) {
    state.rule50_ply = 0;
  } else {
    state.rule50_ply++;
  }
  state.zobrist_hash = 0; // TODO: implement zobrist hashing
  state.history_hash = 0; //boost::hash_combine(state.history_hash, state.zobrist_hash);
  state.recent_hashes.push_back(state.zobrist_hash);
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
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
