#include <core/TrainingTargets.hpp>

#include <core/BasicTypes.hpp>

namespace core {

template<typename Game>
typename PolicyTarget<Game>::Tensor
PolicyTarget<Game>::tensorize(const GameLogView& view) {
  return *view.policy;
}

template <typename Game>
typename ValueTarget<Game>::Tensor ValueTarget<Game>::tensorize(const GameLogView& view) {
  using Rules = Game::Rules;
  seat_index_t cp = Rules::get_current_player(*view.cur_pos);
  Tensor tensor = *view.game_result;
  Game::GameResults::left_rotate(tensor, cp);
  return tensor;
}

template <typename Game>
typename ActionValueTarget<Game>::Tensor ActionValueTarget<Game>::tensorize(
    const GameLogView& view) {
  return *view.action_values;
}

template<typename Game>
typename OppPolicyTarget<Game>::Tensor
OppPolicyTarget<Game>::tensorize(const GameLogView& view) {
  return *view.next_policy;
}

}  // namespace core
