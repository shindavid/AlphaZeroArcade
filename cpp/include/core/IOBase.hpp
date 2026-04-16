#pragma once

#include "core/BasicTypes.hpp"
#include "util/Exceptions.hpp"

#include <boost/json/value.hpp>

#include <string>

namespace core {

template <typename Types>
struct IOBase {
  using Move = Types::Move;
  using State = Types::State;
  using InfoSet = Types::InfoSet;

  static std::string action_delimiter() { return "-"; }
  static std::string player_to_str(core::seat_index_t player) { return std::to_string(player); }
  static void print_state(std::ostream&, const State&, const Move* last_move = nullptr,
                          const Types::player_name_array_t* player_names = nullptr) {
    throw util::CleanException("print_state not implemented");
  }
  static std::string compact_state_repr(const State& state) {
    throw util::CleanException("compact_state_repr not implemented");
  }
  static boost::json::value state_to_json(const State& state) {
    throw util::CleanException("state_to_json not implemented");
  }
  static void add_render_info(const State& state, boost::json::object& obj) {}
};

}  // namespace core
