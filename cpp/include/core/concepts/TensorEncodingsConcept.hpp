#pragma once

#include "core/concepts/GameResultEncodingConcept.hpp"
#include "core/concepts/InputEncoderConcept.hpp"
#include "core/concepts/PolicyEncodingConcept.hpp"

namespace core::concepts {

template <typename TE>
concept TensorEncodings = requires {
  requires core::concepts::PolicyEncoding<typename TE::PolicyEncoding>;
  requires core::concepts::InputEncoder<typename TE::InputEncoder>;
  requires core::concepts::GameResultEncoding<typename TE::GameResultEncoding,
                                              typename TE::InputEncoder::Game>;
};

}  // namespace core::concepts
