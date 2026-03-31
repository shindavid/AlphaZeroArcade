#include "alpha0/NodeStableData.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
NodeStableData<EvalSpec>::NodeStableData(const State& s, int n_valid_moves, core::seat_index_t i)
    : Base(s) {
  R.setZero();  // to be set lazily
  R_valid = false;
  num_valid_moves = n_valid_moves;
  game_phase = Game::Rules::get_game_phase(s);
  is_chance_node = Game::Rules::is_chance_phase(game_phase);
  active_seat = i;
  terminal = false;
}

template <core::concepts::EvalSpec EvalSpec>
NodeStableData<EvalSpec>::NodeStableData(const State& s, const GameResultTensor& game_outcome)
    : Base(s) {
  R = game_outcome;
  R_valid = true;
  num_valid_moves = 0;
  game_phase = -1;
  active_seat = -1;
  terminal = true;
  is_chance_node = false;
}

}  // namespace alpha0
