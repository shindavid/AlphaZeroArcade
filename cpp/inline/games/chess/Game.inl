#include "core/DefaultCanonicalizer.hpp"
#include "games/chess/Game.hpp"
#include "util/AnsiCodes.hpp"
#include "util/BitMapUtil.hpp"
#include "util/BitSet.hpp"
#include "util/CppUtil.hpp"

#include <boost/lexical_cast.hpp>

#include <algorithm>
#include <bit>
#include <iostream>

namespace chess {

inline void Game::Rules::init_state(State& state) {
  // 0, 1 constants match usage in lc0/src/neural/encoder_test.cc
  state = lczero::Position(lczero::ChessBoard::kStartposBoard, 0, 1);
}

inline Game::Types::ActionMask Game::Rules::get_legal_moves(const StateHistory& history) {
  const State& state = history.current();
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

inline void Game::Rules::apply(StateHistory& history, core::action_t action) {
  history.lc0_history().Append(lczero::MoveFromNNIndex(action, 0));
  throw std::runtime_error("Not implemented");
}

inline bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                                     core::action_t last_action, GameResults::Tensor& outcome) {
  throw std::runtime_error("Not implemented");
}

inline std::string Game::IO::action_to_str(core::action_t action, core::action_mode_t) {
  return lczero::MoveFromNNIndex(action, 0).as_string();
}

template <typename Iter>
Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(Iter start, Iter cur) {
  throw std::runtime_error("Not implemented");
}

}  // namespace chess
