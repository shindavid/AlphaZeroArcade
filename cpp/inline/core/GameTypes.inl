#include "core/GameTypes.hpp"
#include "core/BasicTypes.hpp"
#include "util/Exceptions.hpp"

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
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::ActionResponse(
  action_t a) {
  if (a < 0) {
    type_ = kInvalidResponse;
  } else {
    type_ = kMakeMove;
    action_ = a;
  }
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::construct(
  response_type_t type) {
  ActionResponse r;
  r.type_ = type;
  return r;
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::backtrack(
  game_tree_index_t ix) {
  ActionResponse r = construct(kBacktrack);
  r.backtrack_node_ix_ = ix;
  return r;
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::yield(
  int extra_enqueue_count) {
  ActionResponse r = construct(kYieldResponse);
  r.extra_enqueue_count_ = extra_enqueue_count;
  return r;
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
void GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::set_action(
  action_t a) {

  if (a < 0) {
    type_ = kInvalidResponse;
  } else {
    type_ = kMakeMove;
    action_ = a;
  }
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
core::yield_instruction_t
GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionResponse::get_yield_instruction()
  const {
  switch (type_) {
    case kYieldResponse:
      return core::kYield;
    case kDropResponse:
      return core::kDrop;
    default:
      return core::kContinue;
  }
}

template <concepts::GameConstants GameConstants, typename State_, concepts::GameResults GameResults,
          group::concepts::FiniteGroup SymmetryGroup>
bool GameTypes<GameConstants, State_, GameResults, SymmetryGroup>::ActionRequest::permits(
  const ActionResponse& response) const {
  switch (response.type()) {
    case ActionResponse::kMakeMove:
      return valid_actions[response.get_action()];
    case ActionResponse::kUndoLastMove:
      return undo_allowed;
    case ActionResponse::kBacktrack:
      return false;  // backtrack not yet supported
    case ActionResponse::kResignGame:
      return GameConstants::kNumPlayers == 2;
    case ActionResponse::kYieldResponse:
      return true;
    case ActionResponse::kDropResponse:
      return true;
    case ActionResponse::kInvalidResponse:
      return false;
    default:
      throw util::Exception("Unexpected ActionResponse type: {}", response.type());
  }
}


}  // namespace core
