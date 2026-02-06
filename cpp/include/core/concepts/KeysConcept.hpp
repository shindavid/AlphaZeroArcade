#pragma once

#include "util/CppUtil.hpp"

#include <concepts>

namespace core::concepts {

template <typename K, typename State>
concept _KeysHelper = requires(const State& state, typename K::InputTensorizor* input_tensorizor) {
  requires util::concepts::UsableAsHashMapKey<typename K::TransposeKey>;
  requires util::concepts::UsableAsHashMapKey<typename K::EvalKey>;

  { K::transpose_key(state) } -> std::same_as<typename K::TransposeKey>;
  { K::eval_key(input_tensorizor) } -> std::same_as<typename K::EvalKey>;
};

template <typename K, typename Game>
concept Keys = requires { requires _KeysHelper<K, typename Game::State>; };

}  // namespace core::concepts
