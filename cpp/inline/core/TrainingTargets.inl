#include <core/TrainingTargets.hpp>

#include <core/BasicTypes.hpp>

namespace core {

template <typename Game>
bool PolicyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor,
                                   seat_index_t active_seat) {
  if (!view.policy) return false;
  tensor = *view.policy;
  return true;
}

template <typename Game>
bool ValueTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor,
                                  seat_index_t active_seat) {
  tensor = *view.game_result;
  Game::GameResults::left_rotate(tensor, active_seat);
  return true;
}

template <typename Game>
bool ActionValueTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor,
                                        seat_index_t active_seat) {
  if (!view.action_values) return false;
  tensor = *view.action_values;
  return true;
}

template <typename Game>
bool OppPolicyTarget<Game>::tensorize(const GameLogView& view, Tensor& tensor,
                                      seat_index_t active_seat) {
  if (!view.next_policy) return false;
  tensor = *view.next_policy;
  return true;
}

}  // namespace core
