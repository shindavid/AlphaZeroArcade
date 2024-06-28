#pragma once

#include <util/CppUtil.hpp>
#include <util/FiniteGroups.hpp>

#include <cstdint>
#include <functional>
#include <tuple>

namespace core {

using seat_index_t = int8_t;
using player_id_t = int8_t;
using action_t = int32_t;
using action_index_t = int32_t;
using game_id_t = int64_t;
using game_thread_id_t = int16_t;

/*
 * An ActionResponse is an action together with an optional bool indicating whether the player
 * believes their victory is guaranteed.
 *
 * A GameServer can be configured to trust this guarantee, and immediately end the game. This
 * can speed up simulations.
 */
struct ActionResponse {
  ActionResponse() : victory_guarantee(false) {}
  ActionResponse(action_t a, bool v = false) : action(a), victory_guarantee(v) {}

  action_t action;
  bool victory_guarantee;
};

template<typename ValueArray>
struct ActionOutcome {
  ActionOutcome(const ValueArray& value) : terminal_value(value), terminal(true) {}
  ActionOutcome() : terminal(false) {}

  ValueArray terminal_value;  // only valid if terminal
  bool terminal;
};

}  // namespace core
