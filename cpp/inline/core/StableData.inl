#include "core/StableData.hpp"

namespace core {

template <core::concepts::EvalSpec EvalSpec>
StableData<EvalSpec>::StableData(const StateHistory& history, core::seat_index_t as)
    : Base(history.current()) {
  VT.setZero();  // to be set lazily
  VT_valid = false;
  valid_action_mask = Game::Rules::get_legal_moves(history);
  num_valid_actions = valid_action_mask.count();
  action_mode = Game::Rules::get_action_mode(history.current());
  is_chance_node = Game::Rules::is_chance_mode(action_mode);
  active_seat = as;
  terminal = false;
}

template <core::concepts::EvalSpec EvalSpec>
StableData<EvalSpec>::StableData(const StateHistory& history, const ValueTensor& game_outcome)
    : Base(history.current()) {
  VT = game_outcome;
  VT_valid = true;
  num_valid_actions = 0;
  action_mode = -1;
  active_seat = -1;
  terminal = true;
  is_chance_node = false;
}

}  // namespace core
