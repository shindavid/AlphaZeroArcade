#pragma once

#include "core/concepts/GameConcept.hpp"
#include "util/EigenUtil.hpp"

#include <concepts>

namespace core::concepts {

template <typename PE>
concept PolicyEncoding = requires {
  requires core::concepts::Game<typename PE::Game>;
  requires eigen_util::concepts::FTensor<typename PE::Tensor>;
  requires eigen_util::concepts::Shape<typename PE::Shape>;
};

}  // namespace core::concepts
