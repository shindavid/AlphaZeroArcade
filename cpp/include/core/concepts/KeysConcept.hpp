#pragma once

#include "util/CppUtil.hpp"

#include <concepts>

namespace core::concepts {

template <typename K, typename State, typename StateHistory>
concept _KeysHelper = requires(const StateHistory& history, const State* start, const State* cur,
                        std::vector<State>::const_iterator vec_start,
                        std::vector<State>::const_iterator vec_cur) {
  requires util::concepts::UsableAsHashMapKey<typename K::TransposeKey>;
  requires util::concepts::UsableAsHashMapKey<typename K::EvalKey>;

  { K::transpose_key(history) } -> std::same_as<typename K::TransposeKey>;

  // We actually require that K::eval_key() accepts arbitrary random-access iterators of
  // Game::State, but we can't express that directly in the concept. So we do a "poor-man's check"
  // by checking that it works both for raw pointers and for std::vector iterators
  { K::eval_key(start, cur) } -> std::same_as<typename K::EvalKey>;
  { K::eval_key(vec_start, vec_cur) } -> std::same_as<typename K::EvalKey>;
};

template <typename K, typename Game>
concept Keys = requires {
  requires _KeysHelper<K, typename Game::State, typename Game::StateHistory>;
};

}  // namespace core::concepts
