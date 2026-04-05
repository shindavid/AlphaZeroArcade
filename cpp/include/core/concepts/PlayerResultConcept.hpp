#pragma once

#include "core/concepts/AggregationConcept.hpp"

#include <concepts>
#include <string>

namespace core::concepts {

template <class PR>
concept PlayerResult = requires(const PR& r) {
  requires core::concepts::Aggregation<typename PR::Aggregation, PR>;

  { r.to_str() } -> std::same_as<std::string>;
  { r.is_win() } -> std::same_as<bool>;
  { r.is_loss() } -> std::same_as<bool>;
};

}  // namespace core::concepts
