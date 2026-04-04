#include "core/TrainingTargets.hpp"

namespace core {

template <core::concepts::PolicyEncoding PolicyEncoding>
template <typename GameLogView>
bool PolicyTarget<PolicyEncoding>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.policy_valid) return false;
  tensor = view.policy;
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ValueTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor = view.game_result;
  Game::GameResults::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ActionValueTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.action_values_valid) return false;
  tensor = view.action_values;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool QTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor = view.Q;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool QMinTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor = view.Q_min;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool QMaxTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor = view.Q_max;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool WTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor = view.W;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ActionValueUncertaintyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.action_values_valid) return false;
  tensor = view.AU;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::PolicyEncoding PolicyEncoding>
template <typename GameLogView>
bool OppPolicyTarget<PolicyEncoding>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.next_policy_valid) return false;
  tensor = view.next_policy;
  return true;
}

}  // namespace core
