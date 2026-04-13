#include "alpha0/NodeStableData.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
NodeStableData<Spec>::NodeStableData(const State& s, int n_valid_moves, core::seat_index_t i)
    : Base(s) {
  R.setZero();  // to be set lazily
  R_valid = false;
  num_valid_moves = n_valid_moves;
  is_chance_node = Game::Rules::is_chance_state(s);
  active_seat = i;
  terminal = false;
}

template <alpha0::concepts::Spec Spec>
NodeStableData<Spec>::NodeStableData(const State& s, const GameOutcome& game_outcome)
    : Base(s) {
  R = GameResultEncoding::encode(game_outcome);
  R_valid = true;
  num_valid_moves = 0;
  active_seat = -1;
  terminal = true;
  is_chance_node = false;
}

}  // namespace alpha0
