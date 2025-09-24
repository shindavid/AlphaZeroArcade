#include "games/chess/Game.hpp"

#include <boost/lexical_cast.hpp>

namespace chess {

inline void Game::Rules::init_state(State& state) {
  // 0, 1 constants match usage in lc0/src/neural/encoder_test.cc
  state = lczero::Position(lczero::ChessBoard::kStartposBoard, 0, 1);
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  std::vector<lczero::Move> move_list = state.GetBoard().GenerateLegalMoves();
  Types::ActionMask mask;

  for (lczero::Move move : move_list) {
    mask[move.as_nn_index(0)] = true;
  }

  return mask;
}

inline core::seat_index_t Game::Rules::get_current_player(const State& state) {
  return state.IsBlackToMove() ? kBlack : kWhite;
}

inline void Game::Rules::apply(State& state, core::action_t action) {
  throw std::runtime_error("Not implemented");
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
  throw std::runtime_error("Not implemented");
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  return lczero::MoveFromNNIndex(action, 0).as_string();
}

}  // namespace chess
