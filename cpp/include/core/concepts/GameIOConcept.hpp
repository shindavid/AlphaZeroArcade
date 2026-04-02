#pragma once

#include "core/BasicTypes.hpp"

#include <boost/json/value.hpp>

#include <concepts>
#include <string>

namespace core {
namespace concepts {

template <typename GI, typename Move, typename GameTypes>
concept GameIO = requires(std::ostream& ss, const GameTypes::State& state, const Move* last_move,
                          const typename GameTypes::player_name_array_t* player_name_array_ptr) {
  { GI::action_delimiter() } -> std::same_as<std::string>;
  { GI::player_to_str(core::seat_index_t{}) } -> std::same_as<std::string>;
  { GI::print_state(ss, state, last_move, player_name_array_ptr) };
  { GI::print_state(ss, state, last_move) };
  { GI::print_state(ss, state) };

  // compact_state_repr is used in testing and debugging
  // SearchLog requires Game to have implemented state_repr in IO
  // Some MCTS tests require this to be implemented
  { GI::compact_state_repr(state) } -> std::same_as<std::string>;
};

// WebGameIO is a concept that requires GameIO to be implemented, and also requires
// a GI:: state_to_json() method that returns a boost::json::value.
template <typename GI, typename Move, typename GameTypes>
concept WebGameIO = GameIO<GI, Move, GameTypes> && requires(const GameTypes::State& state) {
  { GI::state_to_json(state) } -> std::same_as<boost::json::value>;
};

}  // namespace concepts
}  // namespace core
