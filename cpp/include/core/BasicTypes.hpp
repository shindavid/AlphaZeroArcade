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
using group_id_t = int32_t;

struct symmetry_t {
  bool operator==(const symmetry_t& other) const = default;
  symmetry_t inverse() const { return {group_id, inverse_element, element}; }

  group_id_t group_id = 0;
  group::element_t element = 0;
  group::element_t inverse_element = 0;
};

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

namespace std {

template<>
struct hash<core::symmetry_t> {
  size_t operator()(const core::symmetry_t& sym) const {
    return util::tuple_hash(std::make_tuple(sym.group_id, sym.element));
  }
};

}  // std
