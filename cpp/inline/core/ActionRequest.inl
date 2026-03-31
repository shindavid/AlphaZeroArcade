#include "core/ActionRequest.hpp"

#include "util/Exceptions.hpp"

#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>

namespace core {

template <concepts::Game Game>
bool ActionRequest<Game>::permits(const ActionResponse& response) const {
  switch (response.type()) {
    case ActionResponse::kInvalidResponse:
      return false;
    case ActionResponse::kMakeMove:
      return valid_moves.contains(response.get_move());
    case ActionResponse::kUndoLastMove:
      return undo_allowed;
    case ActionResponse::kBacktrack:
      return true;
    case ActionResponse::kResignGame:
      return GameConstants::kNumPlayers == 2;
    case ActionResponse::kYieldResponse:
      return true;
    case ActionResponse::kForwardRequestRemotely:
      return true;
    case ActionResponse::kDropResponse:
      return true;
    default:
      throw util::Exception("Unexpected ActionResponse type: {}", response.type());
  }
}

}  // namespace core
