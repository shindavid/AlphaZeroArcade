#pragma once

#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"

#include <concepts>

namespace core {
namespace concepts {

template <typename IT, typename State, typename StateHistory>
concept GameInputTensorizor =
  requires(const State& state1, const State& state2, const StateHistory& history) {
    requires eigen_util::concepts::FTensor<typename IT::Tensor>;
    requires util::concepts::UsableAsHashMapKey<typename IT::MCTSKey>;
    requires util::concepts::UsableAsHashMapKey<typename IT::EvalKey>;

    { IT::mcts_key(history) } -> std::same_as<typename IT::MCTSKey>;
    { IT::eval_key(&state1, &state2) } -> std::same_as<typename IT::EvalKey>;
    { IT::tensorize(&state1, &state2) } -> std::same_as<typename IT::Tensor>;
  };

}  // namespace concepts
}  // namespace core
