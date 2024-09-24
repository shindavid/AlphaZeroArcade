#pragma once

#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

#include <concepts>

namespace core {
namespace concepts {

template <typename IT, typename BaseState, typename StateHistory>
concept GameInputTensorizor = requires(const BaseState& base_state1, const BaseState& base_state2,
                                       const StateHistory& history) {
  requires eigen_util::concepts::FTensor<typename IT::Tensor>;
  requires util::concepts::UsableAsHashMapKey<typename IT::MCTSKey>;
  requires util::concepts::UsableAsHashMapKey<typename IT::EvalKey>;

  { IT::mcts_key(history) } -> std::same_as<typename IT::MCTSKey>;
  { IT::eval_key(&base_state1, &base_state2) } -> std::same_as<typename IT::EvalKey>;
  { IT::tensorize(&base_state1, &base_state2) } -> std::same_as<typename IT::Tensor>;
};

}  // namespace concepts
}  // namespace core
