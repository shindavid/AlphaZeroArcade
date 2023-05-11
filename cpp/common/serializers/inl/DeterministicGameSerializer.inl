#include <common/serializers/DeterministicGameSerializer.hpp>

#include <boost/lexical_cast.hpp>

#include <util/Exception.hpp>

namespace common {

template <GameStateConcept GameState>
size_t DeterministicGameSerializer<GameState>::serialize_state_change(
    char* buf, size_t buf_size, const GameState& state, seat_index_t seat, action_index_t action) const {
  return this->serialize_action(buf, buf_size, action);
}

template <GameStateConcept GameState>
void DeterministicGameSerializer<GameState>::deserialize_state_change(
    const char* buf, GameState* state, seat_index_t* seat, action_index_t* action) const {
  *seat = state->get_current_player();
  this->deserialize_action(buf, action);
  state->apply_move(*action);
}

}  // namespace common