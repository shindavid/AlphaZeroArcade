#include <core/TrainingTargets.hpp>

#include <core/BasicTypes.hpp>

namespace core {

template<typename Game, action_type_t ActionType>
typename PolicyTarget<Game, ActionType>::Tensor
PolicyTarget<Game, ActionType>::tensorize(const GameLogView& view) {
  if (view.policy->index() == ActionType) {
    return std::get<ActionType>(*view.policy);
  } else {
    return Tensor::Zero();
  }
}

template <typename Game>
typename ValueTarget<Game>::Tensor ValueTarget<Game>::tensorize(const GameLogView& view) {
  using Rules = Game::Rules;
  seat_index_t cp = Rules::get_current_player(*view.cur_pos);
  Tensor tensor = *view.game_result;
  Game::GameResults::left_rotate(tensor, cp);
  return tensor;
}

template <typename Game, action_type_t ActionType>
typename ActionValueTarget<Game, ActionType>::Tensor
ActionValueTarget<Game, ActionType>::tensorize(
    const GameLogView& view) {
  if (view.action_values->index() == ActionType) {
    return std::get<ActionType>(*view.action_values);
  } else {
    return Tensor::Zero();
  }
}

template<typename Game, action_type_t ActionType>
typename OppPolicyTarget<Game, ActionType>::Tensor
OppPolicyTarget<Game, ActionType>::tensorize(const GameLogView& view) {
  if (view.next_policy->index() == ActionType) {
    return std::get<ActionType>(*view.next_policy);
  } else {
    return Tensor::Zero();
  }
}

}  // namespace core
