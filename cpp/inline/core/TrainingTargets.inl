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
  if (!view.AW_valid) return false;
  tensor = view.AW;
  eigen_util::left_rotate(tensor, view.active_seat);
  return true;
}

template <core::concepts::Game Game>
template <typename GameLogView>
bool ValidActionsTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.policy_valid) return false;
  tensor.setZero();
  auto mask = Game::Rules::get_legal_moves(view.cur_pos);
  for (core::action_t a : mask.on_indices()) {
    tensor(a) = 1.0f;
  }
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
