#include <core/serializers/GeneralSerializer.hpp>

#include <boost/lexical_cast.hpp>

#include <util/Exception.hpp>

namespace core {

template <GameStateConcept GameState>
size_t GeneralSerializer<GameState>::serialize_action_response(
    char* buf, size_t buf_size, const ActionResponse& response) const {
  if (sizeof(response) > buf_size) {
    throw util::Exception("Buffer too small (%ld > %ld)", sizeof(response), buf_size);
  }
  memcpy(buf, &response, sizeof(response));
  return sizeof(response);
}

template <GameStateConcept GameState>
void GeneralSerializer<GameState>::deserialize_action_response(const char* buf,
                                                               ActionResponse* response) const {
  memcpy(response, buf, sizeof(*response));
  GameStateTypes::validate_action(response->action);
}

template <GameStateConcept GameState>
size_t GeneralSerializer<GameState>::serialize_action_prompt(
    char* buf, size_t buf_size, const ActionMask& valid_actions) const {
  return eigen_util::serialize(buf, buf_size, valid_actions);
}

template <GameStateConcept GameState>
void GeneralSerializer<GameState>::deserialize_action_prompt(const char* buf,
                                                             ActionMask* valid_actions) const {
  eigen_util::deserialize(buf, valid_actions);
}

template <GameStateConcept GameState>
size_t GeneralSerializer<GameState>::serialize_state_change(char* buf, size_t buf_size,
                                                            const GameState& state,
                                                            seat_index_t seat,
                                                            const Action& action) const {
  if (sizeof(state) + sizeof(seat) + sizeof(action) > buf_size) {
    throw util::Exception("Buffer too small (%ld + %ld + %ld > %ld)", sizeof(state), sizeof(seat),
                          sizeof(action), buf_size);
  }
  memcpy(buf, &state, sizeof(state));
  memcpy(buf + sizeof(state), &seat, sizeof(seat));
  memcpy(buf + sizeof(state) + sizeof(seat), &action, sizeof(action));
  return sizeof(state) + sizeof(seat) + sizeof(action);
}

template <GameStateConcept GameState>
void GeneralSerializer<GameState>::deserialize_state_change(const char* buf, GameState* state,
                                                            seat_index_t* seat,
                                                            Action* action) const {
  memcpy(state, buf, sizeof(*state));
  memcpy(seat, buf + sizeof(*state), sizeof(*seat));
  memcpy(action, buf + sizeof(*state) + sizeof(*seat), sizeof(*action));
  GameStateTypes::validate_action(*action);
}

template <GameStateConcept GameState>
size_t GeneralSerializer<GameState>::serialize_game_end(char* buf, size_t buf_size,
                                                        const GameOutcome& outcome) const {
  if (sizeof(outcome) > buf_size) {
    throw util::Exception("Buffer too small (%ld > %ld)", sizeof(outcome), buf_size);
  }
  memcpy(buf, &outcome, sizeof(outcome));
  return sizeof(outcome);
}

template <GameStateConcept GameState>
void GeneralSerializer<GameState>::deserialize_game_end(const char* buf,
                                                        GameOutcome* outcome) const {
  *outcome = reinterpret_cast<const GameOutcome&>(*buf);
}

}  // namespace core
