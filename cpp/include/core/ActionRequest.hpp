#pragma once

#include "core/ActionResponse.hpp"
#include "core/GameDerivedConstants.hpp"
#include "core/StateIterator.hpp"
#include "core/YieldManager.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <concepts::Game Game>
struct ActionRequest {
  using GameConstants = Game::Constants;
  using State = Game::State;
  using VerboseDataIterator = core::VerboseDataIterator<Game>;

  static constexpr int kMaxNumActions = DerivedConstants<GameConstants>::kMaxNumActions;

  using ActionMask = util::CompactBitSet<kMaxNumActions>;

  ActionRequest(const State& s, const ActionMask& va, const YieldNotificationUnit& u,
                game_tree_node_aux_t a, VerboseDataIterator i)
      : state(s), valid_actions(va), notification_unit(u), verbose_data_iterator(i), aux(a) {}

  ActionRequest(const State& s, const ActionMask& va) : state(s), valid_actions(va) {}

  bool permits(const ActionResponse& response) const;

  const State& state;
  const ActionMask& valid_actions;
  YieldNotificationUnit notification_unit;
  VerboseDataIterator verbose_data_iterator;
  game_tree_node_aux_t aux = 0;

  // If set to true, the player is being asked to play noisily, in order to add opening diversity.
  // Each player is free to interpret this in their own way.
  bool play_noisily = false;
  bool undo_allowed = false;
};

}  // namespace core

#include "inline/core/ActionRequest.inl"
