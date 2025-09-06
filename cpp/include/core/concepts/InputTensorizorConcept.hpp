#pragma once

#include "util/EigenUtil.hpp"

#include <concepts>
#include <vector>

namespace core::concepts {

template <typename IT, typename State>
concept InputTensorizor =
  requires(const State* start, const State* cur, std::vector<State>::const_iterator vec_start,
           std::vector<State>::const_iterator vec_cur) {
    requires eigen_util::concepts::FTensor<typename IT::Tensor>;

    // We actually require that IT::tensorize() accepts arbitrary random-access iterators of
    // State, but we can't express that directly in the concept. So we do a "poor-man's check"
    // by checking that it works both for raw pointers and for std::vector iterators
    { IT::tensorize(start, cur) } -> std::same_as<typename IT::Tensor>;
    { IT::tensorize(vec_start, vec_cur) } -> std::same_as<typename IT::Tensor>;
  };

}  // namespace core::concepts
