#pragma once

#include "core/BasicTypes.hpp"

#include <cstdint>

namespace core {

/*
 * An ActionResponse is a player's response to an ActionRequest. Typically, it is simply a
 * core::action_t specifying the action to take. However, there are some other possible responses:
 *
 * - undo last move
 * - backtrack to a previous game-tree node
 * - resign the game
 * - yield (i.e., ask for more thinking time)
 * - drop (i.e., drop this auxiliary thread / context)
 *
 * See GameServer.hpp for a discussion of how yielding and dropping work.
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
