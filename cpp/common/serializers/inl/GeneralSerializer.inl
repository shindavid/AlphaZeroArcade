#include <common/serializers/GeneralSerializer.hpp>

#include <boost/lexical_cast.hpp>

#include <util/Exception.hpp>

namespace common {

template <GameStateConcept GameState>
size_t GeneralSerializer<GameState>::serialize_action(char* buf, size_t buf_size, action_index_t action) const {
  size_t n = snprintf(buf, buf_size, "%d", action);
  if (n >= buf_size) {
    throw util::Exception("Buffer too small (%ld >= %ld)", n, buf_size);
  }
  return n;
}

template <GameStateConcept GameState>
void GeneralSerializer<GameState>::deserialize_action(const char* buf, action_index_t* action) const {
  *action = boost::lexical_cast<action_index_t>(buf);

  if (*action < 0 || *action >= GameStateTypes::kNumGlobalActions) {
    throw util::Exception("Invalid action \"%s\" (action=%d)", buf, *action);
  }
}

template <GameStateConcept GameState>
size_t GeneralSerializer<GameState>::serialize_action_prompt(
    char* buf, size_t buf_size, const ActionMask& valid_actions) const {
  if (sizeof(valid_actions) > buf_size) {
    throw util::Exception("Buffer too small (%ld > %ld)", sizeof(valid_actions), buf_size);
  }
  memcpy(buf, &valid_actions, sizeof(valid_actions));
  return sizeof(valid_actions);
}

template <GameStateConcept GameState>
void GeneralSerializer<GameState>::deserialize_action_prompt(
    const char* buf, ActionMask* valid_actions) const {
  memcpy(valid_actions, buf, sizeof(*valid_actions));
}

template <GameStateConcept GameState>
size_t GeneralSerializer<GameState>::serialize_state_change(
    char* buf, size_t buf_size, const GameState& state, seat_index_t seat, action_index_t action) const {
  if (sizeof(state) + sizeof(seat) + sizeof(action) > buf_size) {
    throw util::Exception("Buffer too small (%ld + %ld + %ld > %ld)",
                          sizeof(state), sizeof(seat), sizeof(action), buf_size);
  }
  memcpy(buf, &state, sizeof(state));
  memcpy(buf + sizeof(state), &seat, sizeof(seat));
  memcpy(buf + sizeof(state) + sizeof(seat), &action, sizeof(action));
  return sizeof(state) + sizeof(seat) + sizeof(action);
}

template <GameStateConcept GameState>
void GeneralSerializer<GameState>::deserialize_state_change(
    const char* buf, GameState* state, seat_index_t* seat, action_index_t* action) const {
  memcpy(state, buf, sizeof(*state));
  memcpy(seat, buf + sizeof(*state), sizeof(*seat));
  memcpy(action, buf + sizeof(*state) + sizeof(*seat), sizeof(*action));
}

template <GameStateConcept GameState>
size_t GeneralSerializer<GameState>::serialize_game_end(char* buf, size_t buf_size, const GameOutcome& outcome) const {
  if (sizeof(outcome) > buf_size) {
    throw util::Exception("Buffer too small (%ld > %ld)", sizeof(outcome), buf_size);
  }
  memcpy(buf, &outcome, sizeof(outcome));
  return sizeof(outcome);
}

template <GameStateConcept GameState>
void GeneralSerializer<GameState>::deserialize_game_end(const char* buf, GameOutcome* outcome) const {
  *outcome = reinterpret_cast<const GameOutcome&>(*buf);
}

}  // namespace common
