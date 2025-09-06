#pragma once

#include "util/CppUtil.hpp"

#include <concepts>

namespace core::concepts {

template <typename K, typename G>
concept Keys =
  requires(const typename G::StateHistory& history, const typename G::State* start,
           const typename G::State* cur, std::vector<typename G::State>::const_iterator vec_start,
           std::vector<typename G::State>::const_iterator vec_cur) {
    requires util::concepts::UsableAsHashMapKey<typename K::TransposeKey>;
    requires util::concepts::UsableAsHashMapKey<typename K::EvalKey>;

    { K::transpose_key(history) } -> std::same_as<typename K::TransposeKey>;

    // We actually require that K::eval_key() accepts arbitrary random-access iterators of
    // Game::State, but we can't express that directly in the concept. So we do a "poor-man's check"
    // by checking that it works both for raw pointers and for std::vector iterators
    { K::eval_key(start, cur) } -> std::same_as<typename K::EvalKey>;
    { K::eval_key(vec_start, vec_cur) } -> std::same_as<typename K::EvalKey>;
  };

}  // namespace core::concepts
