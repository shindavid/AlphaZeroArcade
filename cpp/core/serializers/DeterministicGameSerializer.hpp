#pragma once

#include <core/GameStateConcept.hpp>
#include <core/serializers/GeneralSerializer.hpp>

namespace core {

/*
 * The DeterministicGameSerializer is identical to the GeneralSerializer, except that it assumes that the underlying
 * game is deterministic. This allows us to optimize the state-change serialization/deserialization methods - rather
 * than sending the entire GameState, we just send the action, and reconstruct the GameState on the other end. Note that
 * this assumes that the benefit of sending fewer bytes outweighs the cost of reconstructing the GameState.
 */
template <GameStateConcept GameState>
class DeterministicGameSerializer : public GeneralSerializer<GameState> {
public:
  using GameStateTypes = core::GameStateTypes<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;

  size_t serialize_state_change(char* buf, size_t buf_size, const GameState& state, seat_index_t seat, action_t action) const override;
  void deserialize_state_change(const char* buf, GameState* state, seat_index_t* seat, action_t* action) const override;
};

}  // namespace core

#include <core/serializers/inl/DeterministicGameSerializer.inl>
