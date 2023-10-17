#include <core/serializers/DeterministicGameSerializer.hpp>

#include <boost/lexical_cast.hpp>

#include <util/Exception.hpp>

namespace core {

template <GameStateConcept GameState>
size_t DeterministicGameSerializer<GameState>::serialize_state_change(char* buf, size_t buf_size,
                                                                      const GameState& state,
                                                                      seat_index_t seat,
                                                                      const Action& action) const {
  ActionResponse response(action);
  return this->serialize_action_response(buf, buf_size, response);
}

template <GameStateConcept GameState>
void DeterministicGameSerializer<GameState>::deserialize_state_change(const char* buf,
                                                                      GameState* state,
                                                                      seat_index_t* seat,
                                                                      Action* action) const {
  ActionResponse response;
  *seat = state->get_current_player();
  this->deserialize_action_response(buf, &response);
  *action = response.action;
  state->apply_move(*action);
}

}  // namespace core
