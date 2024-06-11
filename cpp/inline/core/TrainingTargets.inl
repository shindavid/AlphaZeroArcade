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
  using Rules = typename Game::Rules;
  seat_index_t cp = Rules::current_player(*view.cur_pos);
  ValueArray shifted_outcome = *view.outcome;
  eigen_util::left_rotate(shifted_outcome, cp);
  return eigen_util::reinterpret_as_tensor<Tensor>(shifted_outcome);
}

template<typename Game>
typename OppPolicyTarget<Game>::Tensor
OppPolicyTarget<Game>::tensorize(const GameLogView& view) {
  return *view.next_policy;
}

}  // namespace core
