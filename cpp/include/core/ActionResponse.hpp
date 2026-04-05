#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

#include <cstdint>
#include <optional>

namespace core {

/*
 * An ActionResponse is a player's response to an ActionRequest. Typically, it is simply a
 * Game::Move specifying the action to take. However, there are some other possible responses:
 *
 * - undo last move
 * - backtrack to a previous game-tree node
 * - resign the game
 * - yield (i.e., ask for more thinking time)
 * - drop (i.e., drop this auxiliary thread / context)
 *
 * See GameServer.hpp for a discussion of how yielding and dropping work.
 */
template <concepts::Game Game>
struct ActionResponse {
  using Move = Game::Move;
  using GameOutcome = Game::Types::GameOutcome;

  enum response_type_t : uint8_t {
    kInvalidResponse,
    kMakeMove,
    kUndoLastMove,
    kBacktrack,
    kResignGame,
    kYieldResponse,
    kForwardRequestRemotely,  // special value used by RemotePlayerProxy
    kDropResponse
  };

  ActionResponse() = default;
  ActionResponse(const Move& move);  // if move is default, kValidResponse, else kMakeMove

  static ActionResponse yield(int extra_enqueue_count = 0);
  static ActionResponse drop() { return construct(kDropResponse); }
  static ActionResponse resign() { return construct(kResignGame); }
  static ActionResponse undo() { return construct(kUndoLastMove); }
  static ActionResponse invalid() { return construct(kInvalidResponse); }
  static ActionResponse backtrack(game_tree_index_t ix);
  static ActionResponse forward_request_remotely() { return construct(kForwardRequestRemotely); }

  template <typename T>
  void set_aux(T aux);

  bool is_aux_set() const { return aux_set_; }
  game_tree_node_aux_t aux() const { return aux_; }
  response_type_t type() const { return type_; }
  void set_move(const Move& move);
  const Move& get_move() const { return move_; }
  core::yield_instruction_t get_yield_instruction() const;
  void set_outcome_guarantee(const GameOutcome& outcome) { outcome_guarantee_ = outcome; }
  const std::optional<GameOutcome>& get_outcome_guarantee() const { return outcome_guarantee_; }
  int get_extra_enqueue_count() const { return extra_enqueue_count_; }
  game_tree_index_t backtrack_node_index() const { return backtrack_node_ix_; }

 private:
  static ActionResponse construct(response_type_t type);

  Move move_;
  game_tree_index_t backtrack_node_ix_ = kNullNodeIx;
  game_tree_node_aux_t aux_ = 0;
  int extra_enqueue_count_ = 0;
  std::optional<GameOutcome> outcome_guarantee_;
  response_type_t type_ = kInvalidResponse;
  bool aux_set_ = false;
};

}  // namespace core

#include "inline/core/ActionResponse.inl"
