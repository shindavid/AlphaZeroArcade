#include "alphazero/NodeStableData.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
NodeStableData<EvalSpec>::NodeStableData(const State& st, core::seat_index_t as) : Base(st) {
  R.setZero();  // to be set lazily
  VT_valid = false;
  valid_action_mask = Game::Rules::get_legal_moves(st);
  num_valid_actions = valid_action_mask.count();
  action_mode = Game::Rules::get_action_mode(st);
  is_chance_node = Game::Rules::is_chance_mode(action_mode);
  active_seat = as;
  terminal = false;
}

template <core::concepts::EvalSpec EvalSpec>
NodeStableData<EvalSpec>::NodeStableData(const State& st, const GameResultTensor& game_outcome)
    : Base(st) {
  R = game_outcome;
  VT_valid = true;
  num_valid_actions = 0;
  action_mode = -1;
  active_seat = -1;
  terminal = true;
  is_chance_node = false;
}

}  // namespace alpha0
