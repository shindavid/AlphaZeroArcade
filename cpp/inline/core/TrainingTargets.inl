#include <core/TrainingTargets.hpp>

#include <core/BasicTypes.hpp>

namespace core {

template<typename Game>
bool PolicyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.policy) return false;
  tensor = *view.policy;
  return true;
}

template <typename Game>
bool ValueTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  using Rules = Game::Rules;
  seat_index_t cp = Rules::get_current_player(*view.cur_pos);
  tensor = *view.game_result;
  Game::GameResults::left_rotate(tensor, cp);
  return true;
}

template <typename Game>
bool ActionValueTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.action_values) return false;
  tensor = *view.action_values;
  return true;
}

template<typename Game>
bool OppPolicyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor) {
  if (!view.next_policy) return false;
  tensor = *view.next_policy;
  return true;
}

}  // namespace core
