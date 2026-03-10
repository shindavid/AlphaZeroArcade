#pragma once

#include "util/CppUtil.hpp"

#include <concepts>

namespace core::concepts {

template <typename T, typename State>
concept Transposer = requires(const State& state) {
  requires util::concepts::UsableAsHashMapKey<typename T::Key>;

  { T::key(state) } -> std::same_as<typename T::Key>;
};

}  // namespace core::concepts
