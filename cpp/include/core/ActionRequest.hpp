#pragma once

#include "core/ActionResponse.hpp"
#include "core/YieldManager.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
struct ActionRequest {
  using GameConstants = Game::Constants;
  using InfoSet = Game::InfoSet;
  using MoveSet = Game::MoveSet;
  using ActionResponse = core::ActionResponse<Game>;

  ActionRequest(const InfoSet& is, const MoveSet& ms, const YieldNotificationUnit& u,
                game_tree_node_aux_t a)
      : info_set(is), valid_moves(ms), notification_unit(u), aux(a) {}

  ActionRequest(const InfoSet& is, const MoveSet& ms) : info_set(is), valid_moves(ms) {}

  bool permits(const ActionResponse& response) const;

  const InfoSet& info_set;
  const MoveSet& valid_moves;
  YieldNotificationUnit notification_unit;
  game_tree_node_aux_t aux = 0;

  // If set to true, the player is being asked to play noisily, in order to add opening diversity.
  // Each player is free to interpret this in their own way.
  bool play_noisily = false;
  bool undo_allowed = false;
};

}  // namespace core

#include "inline/core/ActionRequest.inl"
