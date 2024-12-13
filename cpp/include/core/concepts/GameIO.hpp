#pragma once

#include <core/BasicTypes.hpp>

#include <concepts>
#include <sstream>
#include <string>

namespace core {
namespace concepts {

template <typename GI, typename GameTypes, typename State>
concept GameIO = requires(std::ostream& ss, const State& state,
                          const typename GameTypes::player_name_array_t* player_name_array_ptr,
                          const typename GameTypes::PolicyTensor& policy_tensor,
                          const typename GameTypes::SearchResults& search_results) {
  { GI::action_delimiter() } -> std::same_as<std::string>;
  { GI::action_to_str(core::action_t{}, core::action_mode_t{}) } -> std::same_as<std::string>;
  { GI::print_state(ss, state, core::action_t{}, player_name_array_ptr) };
  { GI::print_state(ss, state, core::action_t{}) };
  { GI::print_state(ss, state) };
  { GI::print_mcts_results(ss, policy_tensor, search_results) };
  // compact_state_repr is used in testing and debugging
  // SearchLog requires Game to have implemented state_repr in IO
  // Some MCTS tests require this to be implemented
  { GI::compact_state_repr(state) } -> std::same_as<std::string>;
};

}  // namespace concepts
}  // namespace core
