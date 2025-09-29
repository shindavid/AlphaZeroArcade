#include "core/TrainingTargets.hpp"

namespace core {

template <core::concepts::Game Game>
template <typename GameLogView>
bool PolicyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
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
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool QPosteriorTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  tensor = view.Q_posterior;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ActionValueUncertaintyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.action_value_uncertainties_valid) return false;
  tensor = view.action_value_uncertainties;
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool OppPolicyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.next_policy_valid) return false;
  tensor = view.next_policy;
  return true;
}

}  // namespace core
