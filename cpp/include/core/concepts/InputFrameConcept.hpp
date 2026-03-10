#pragma once

#include <concepts>
#include <type_traits>

namespace core::concepts {

template <typename IF, typename State>
concept InputFrame = requires(const State& state) {
  requires std::is_trivial_v<IF>;
  requires std::is_standard_layout_v<IF>;

  { IF(state) } -> std::same_as<IF>;  // constructible from State
};

}  // namespace core::concepts
