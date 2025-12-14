#include "core/GameTypes.hpp"

namespace core {
template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
void GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::set_aux(
  node_aux_t aux) {
  aux_ = aux;
  aux_set_ = true;
}

}  // namespace core
