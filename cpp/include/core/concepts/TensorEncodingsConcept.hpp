#pragma once

#include "core/concepts/PolicyEncodingConcept.hpp"

#include <concepts>

namespace core::concepts {

template <typename TE>
concept TensorEncodings = requires {
  requires core::concepts::PolicyEncoding<typename TE::PolicyEncoding>;
  typename TE::InputEncoder;
};

}  // namespace core::concepts
