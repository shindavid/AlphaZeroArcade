#include "beta0/NodeStableData.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
NodeStableData<Spec>::NodeStableData(const State& s, int n_valid_moves, core::seat_index_t i)
    : Base(s, n_valid_moves, i) {
  uncertainty_.setZero();
  backup_accu_static.setZero();
}

template <beta0::concepts::Spec Spec>
NodeStableData<Spec>::NodeStableData(const State& s, const GameOutcome& game_outcome)
    : Base(s, game_outcome) {
  uncertainty_.setZero();
  backup_accu_static.setZero();
}

}  // namespace beta0
