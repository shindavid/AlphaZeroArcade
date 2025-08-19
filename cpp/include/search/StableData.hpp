#pragma once

#include "search/StableDataBase.hpp"
#include "search/concepts/Traits.hpp"

namespace search {

// StableData consists of data members of search::NodeBase<Traits> whose values do not change once
// the node is created. The StableData member of search::NodeBase<Traits> is const in spirit, but
// because we have some Node-copying in LookupTable, we cannot make it truly const.
template <typename Traits>
struct StableData : public StableDataBaseImpl<typename Traits::Game> {
  using Game = Traits::Game;
  using Base = StableDataBaseImpl<Game>;
  using StateHistory = Game::StateHistory;
  using ActionMask = Game::Types::ActionMask;
  using ValueTensor = Game::Types::ValueTensor;

  StableData(const StateHistory&, core::seat_index_t active_seat);   // for non-terminal nodes
  StableData(const StateHistory&, const ValueTensor& game_outcome);  // for terminal nodes

  ValueTensor VT;
  ActionMask valid_action_mask;
  int num_valid_actions;
  core::action_mode_t action_mode;

  // active_seat is usually the current player, who is about to make a move
  // if this is a chance node, active_seat is the player who just made a move
  core::seat_index_t active_seat;

  bool terminal;
  bool VT_valid;
  bool is_chance_node;
};

}  // namespace search

#include "inline/search/StableData.inl"
