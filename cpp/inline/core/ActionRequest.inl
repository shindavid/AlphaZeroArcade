#include "core/ActionRequest.hpp"

#include "util/Exceptions.hpp"

#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>

namespace core {

template <concepts::Game Game>
bool ActionRequest<Game>::permits(const ActionResponse& response) const {
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
