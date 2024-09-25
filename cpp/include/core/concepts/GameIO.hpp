#pragma once

#include <core/BasicTypes.hpp>

#include <concepts>
#include <sstream>
#include <string>

namespace core {
namespace concepts {

template <typename GI, typename GameTypes, typename State>
concept GameIO = requires(std::ostream& ss, const State& base_state,
                          const typename GameTypes::player_name_array_t* player_name_array_ptr,
                          const typename GameTypes::PolicyTensor& policy_tensor,
                          const typename GameTypes::SearchResults& search_results) {
  { GI::action_delimiter() } -> std::same_as<std::string>;
  { GI::action_to_str(core::action_t{}) } -> std::same_as<std::string>;
  { GI::print_state(ss, base_state, core::action_t{}, player_name_array_ptr) };
  { GI::print_state(ss, base_state, core::action_t{}) };
  { GI::print_state(ss, base_state) };
  { GI::print_mcts_results(ss, policy_tensor, search_results) };
};

}  // namespace concepts
}  // namespace core
