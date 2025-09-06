#pragma once

#include "core/concepts/KeysConcept.hpp"
#include "util/EigenUtil.hpp"

#include <concepts>
#include <vector>

namespace core::concepts {

template <typename IT, typename Game, typename State>
concept _InputTensorizorHelper =
  requires(const State* start, const State* cur, std::vector<State>::const_iterator vec_start,
           std::vector<State>::const_iterator vec_cur) {
    typename IT::Tensor;
    typename IT::Keys;

    requires eigen_util::concepts::FTensor<typename IT::Tensor>;
    requires core::concepts::Keys<typename IT::Keys, Game>;

    // We actually require that IT::tensorize() accepts arbitrary random-access iterators of
    // State, but we can't express that directly in the concept. So we do a "poor-man's check"
    // by checking that it works both for raw pointers and for std::vector iterators
    { IT::tensorize(start, cur) } -> std::same_as<typename IT::Tensor>;
    { IT::tensorize(vec_start, vec_cur) } -> std::same_as<typename IT::Tensor>;
  };

template <typename K, typename Game>
concept InputTensorizor =
  requires { requires _InputTensorizorHelper<K, Game, typename Game::State>; };

}  // namespace core::concepts
