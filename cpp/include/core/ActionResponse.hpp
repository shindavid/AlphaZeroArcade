#pragma once

#include "core/BasicTypes.hpp"

#include <cstdint>

namespace core {

/*
 * An ActionResponse is an action together with some optional auxiliary information:
 *
 * - victory_guarantee: whether the player believes their victory is guaranteed. GameServer can be
 *     configured to trust this guarantee, and immediately end the game. This can speed up
 *     simulations.
 *
 * - training_info: used to generate targets for NN training.
 *
 * - yield_instruction: Indicates whether the player needs more time to think asynchronously. If
 *     set to a non-kContinue value, then the action/training_info/victory_guarantee fields are
 *     ignored. If set to kDrop, this indicates that this was an auxiliary thread launched for
 *     multithreaded search, and that the multithreaded part is over.
 *
 * - extra_enqueue_count: If set to a nonzero value, this instructs the GameServer to enqueue the
 *     current GameSlot this many additional times. This is useful for players that want to
 *     engage in multithreaded search. This should only be used for instruction type kYield.
 */
struct ActionResponse {
  enum response_type_t : uint8_t {
    kInvalidResponse,
    kMakeMove,
    kUndoLastMove,
    kBacktrack,
    kResignGame,
    kYieldResponse,
    kDropResponse
  };

  // Construct a kMakeMove response if action >= 0; otherwise, kInvalidResponse
  ActionResponse(action_t a = kNullAction);

  static ActionResponse yield(int extra_enqueue_count = 0);
  static ActionResponse drop() { return construct(kDropResponse); }
  static ActionResponse resign() { return construct(kResignGame); }
  static ActionResponse undo() { return construct(kUndoLastMove); }
  static ActionResponse invalid() { return construct(kInvalidResponse); }
  static ActionResponse backtrack(game_tree_index_t ix);

  template <typename T>
  void set_aux(T aux);

  bool is_aux_set() const { return aux_set_; }
  game_tree_node_aux_t aux() const { return aux_; }
  response_type_t type() const { return type_; }
  void set_action(action_t a);
  action_t get_action() const { return action_; }
  core::yield_instruction_t get_yield_instruction() const;
  void set_victory_guarantee(bool v) { victory_guarantee_ = v; }
  bool get_victory_guarantee() const { return victory_guarantee_; }
  int get_extra_enqueue_count() const { return extra_enqueue_count_; }

 private:
  static ActionResponse construct(response_type_t type);

  action_t action_ = kNullAction;
  game_tree_index_t backtrack_node_ix_ = kNullNodeIx;
  game_tree_node_aux_t aux_ = 0;
  int extra_enqueue_count_ = 0;
  bool victory_guarantee_ = false;
  response_type_t type_ = kInvalidResponse;
  bool aux_set_ = false;
};

}  // namespace core

#include "inline/core/ActionResponse.inl"
