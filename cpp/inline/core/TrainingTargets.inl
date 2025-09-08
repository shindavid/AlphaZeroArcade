#include "core/TrainingTargets.hpp"

namespace core {

template <core::concepts::Game Game>
bool PolicyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.policy) return false;
  tensor = *view.policy;
  return true;
}

template <core::concepts::Game Game>
template <typename Dst>
void PolicyTarget<Game>::uniform_init(const ActionMask& valid_actions, Dst& dst) {
  dst.setConstant(1.0 / valid_actions.count());
}

template <core::concepts::Game Game>
bool ValueTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor = *view.game_result;
  Game::GameResults::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename Dst>
void ValueTarget<Game>::uniform_init(const ActionMask& valid_actions, Dst& dst) {
  dst.setConstant(1.0 / dst.size());
}

template <core::concepts::Game Game>
bool ActionValueTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.action_values) return false;
  tensor = *view.action_values;
  return true;
}

template <core::concepts::Game Game>
template <typename Dst>
void ActionValueTarget<Game>::uniform_init(const ActionMask& valid_actions, Dst& dst) {
  dst.setConstant(1.0 / Game::Constants::kNumPlayers);
}

template <core::concepts::Game Game>
bool OppPolicyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.next_policy) return false;
  tensor = *view.next_policy;
  return true;
}

}  // namespace core
