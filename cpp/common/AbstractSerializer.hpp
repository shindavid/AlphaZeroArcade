#pragma once

#include <concepts>

#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>

namespace common {

/*
 * A Serializer is responsible for the serialization and deserialization of GameState changes, actions, and outcomes.
 * It is needed when using a GameServer with remote players.
 *
 * Technically we could make this a concept and avoid virtual methods. But then the interface requirement would become
 * harder to grok, as c++ concepts are not really optimized for readability. Besides, the serializer usages are such
 * that we should not actually incur the virtual overhead (and even if we do, it should be negligible).
 */
template <GameStateConcept GameState>
class AbstractSerializer {
public:
  using GameStateTypes = common::GameStateTypes<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;

  virtual ~AbstractSerializer() = default;

  virtual size_t serialize_action(char* buf, size_t buf_size, action_index_t action) const = 0;
  virtual void deserialize_action(const char* buf, action_index_t* action) const = 0;

  virtual size_t serialize_action_prompt(char* buffer, size_t buffer_size, const ActionMask& valid_actions) const = 0;
  virtual void deserialize_action_prompt(const char* buffer, ActionMask* valid_actions) const = 0;

  virtual size_t serialize_state_change(char* buf, size_t buf_size, const GameState& state, seat_index_t seat,
                                        action_index_t action) const = 0;
  virtual void deserialize_state_change(const char* buf, GameState* state, seat_index_t* seat,
                                        action_index_t* action) const = 0;

  virtual size_t serialize_game_end(char* buffer, size_t buffer_size, const GameOutcome& outcome) const = 0;
  virtual void deserialize_game_end(const char* buffer, GameOutcome* outcome) const = 0;
};

}  // namespace common
