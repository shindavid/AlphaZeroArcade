#pragma once

#include <core/AbstractSerializer.hpp>
#include <core/GameStateConcept.hpp>

namespace common {

/*
 * The GeneralSerializer relies on memcpy to serialize and deserialize. This is simple and does the job. However,
 * there are some notable drawbacks:
 * 
 * - Portability: the serialized data may not be portable across different architectures or compilers.
 * - Robustness: the serialized data may not be robust to changes in the GameState implementation. This may prevent us
 *   from upgrading the GameState implementation without breaking compatibility with old binaries.
 * - Traffic size: the serialized data may be larger than necessary, which may increase network traffic. For example,
 *   in deterministic games, we only need to send the action, not the entire GameState, since the GameState can be
 *   reconstructed from the action.
 */
template <GameStateConcept GameState>
class GeneralSerializer : public AbstractSerializer<GameState> {
public:
  using GameStateTypes = common::GameStateTypes<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;

  size_t serialize_action(char* buf, size_t buf_size, action_index_t action) const override;
  void deserialize_action(const char* buf, action_index_t* action) const override;

  size_t serialize_action_prompt(char* buf, size_t buf_size, const ActionMask& valid_actions) const override;
  void deserialize_action_prompt(const char* buf, ActionMask* valid_actions) const override;

  size_t serialize_state_change(char* buf, size_t buf_size, const GameState& state, seat_index_t seat, action_index_t action) const override;
  void deserialize_state_change(const char* buf, GameState* state, seat_index_t* seat, action_index_t* action) const override;

  size_t serialize_game_end(char* buf, size_t buf_size, const GameOutcome& outcome) const override;
  void deserialize_game_end(const char* buf, GameOutcome* outcome) const override;
};

}  // namespace common

#include <core/serializers/inl/GeneralSerializer.inl>
