#include "core/GameTypes.hpp"
#include "core/BasicTypes.hpp"

namespace core {

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
template <typename T>
void GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::set_aux(T aux) {
  if constexpr (std::is_pointer_v<T>) {
    aux_ = reinterpret_cast<game_tree_node_aux_t>(aux);
  } else {
    // We are being explicit about the supported aux representations. Other types (e.g. char, structs)
    // could be supported in the future, but are intentionally disallowed for now.
    RELEASE_ASSERT(std::is_integral_v<T>,
                   "only integral and pointer types are supported for aux for now");
    aux_ = static_cast<game_tree_node_aux_t>(aux);
  }

  aux_set_ = true;
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::backtrack(
  game_tree_index_t ix) {
  ActionResponse r;
  r.backtrack_node_ix_ = ix;
  r.type_ = kBacktrack;
  return r;
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::make_move(
  action_t a) {
  ActionResponse r(a);
  r.type_ = kMakeMove;
  return r;
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::yield(int e) {
  ActionResponse r(kNullAction, e, core::kYield);
  r.type_ = kYieldResponse;
  return r;
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::drop() {
  ActionResponse r(kNullAction, 0, core::kDrop);
  r.type_ = kDropResponse;
  return r;
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
bool GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionRequest::permits(
  const ActionResponse& response) const {
  switch (response.type()) {
    case ActionResponse::kInvalidResponse:
      return false;
    case ActionResponse::kUndoLastMove:
      return undo_allowed;
    case ActionResponse::kMakeMove:
      return valid_actions[response.action];
    default:
      return true;
  }
}


}  // namespace core
