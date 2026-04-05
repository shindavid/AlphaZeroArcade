#pragma once

#include <concepts>
#include <string>

namespace core::concepts {

template <class A, class PlayerResult>
concept Aggregation = requires(A& agg, const PlayerResult& r) {
  { agg.add(r) } -> std::same_as<void>;
  { agg.to_str() } -> std::same_as<std::string>;
};

}  // namespace core::concepts
