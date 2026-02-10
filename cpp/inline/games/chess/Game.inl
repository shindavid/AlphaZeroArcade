#include "games/chess/Game.hpp"
#include "lc0/chess/board.h"

#include <boost/lexical_cast.hpp>

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

  throw std::runtime_error("Not implemented");
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.seat;
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  throw std::runtime_error("Not implemented");
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
  throw std::runtime_error("Not implemented");
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  throw std::runtime_error("Not implemented");
}

}  // namespace chess
