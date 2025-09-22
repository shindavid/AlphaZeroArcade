#pragma once

#include "core/concepts/KeysConcept.hpp"
#include "util/CppUtil.hpp"
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

    // kNumStatesToEncode is the number of State's that are needed to tensorize a given state. If
    // the neural network does not need any previous State's, kNumStatesToEncode should be 1.
    { util::decay_copy(IT::kNumStatesToEncode) } -> std::same_as<int>;

    // We actually require that IT::tensorize() accepts arbitrary random-access iterators of
    // State, but we can't express that directly in the concept. So we do a "poor-man's check"
    // by checking that it works both for raw pointers and for std::vector iterators
    //
    // We should always have start + kNumStatesToEncode == cur + 1
    { IT::tensorize(start, cur) } -> std::same_as<typename IT::Tensor>;
    { IT::tensorize(vec_start, vec_cur) } -> std::same_as<typename IT::Tensor>;
  };

template <typename K, typename Game>
concept InputTensorizor =
  requires { requires _InputTensorizorHelper<K, Game, typename Game::State>; };

}  // namespace core::concepts
