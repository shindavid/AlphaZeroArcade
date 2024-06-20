#pragma once

#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

#include <concepts>

namespace core {
namespace concepts {

template <typename InputTensorizor, typename BaseState, typename FullState>
concept GameInputTensorizor = requires(const BaseState& base_state1, const BaseState& base_state2,
                                       const FullState& full_state) {
  requires eigen_util::concepts::FTensor<typename InputTensorizor::Tensor>;
  requires util::concepts::UsableAsHashMapKey<typename InputTensorizor::EvalKey>;
  requires util::concepts::UsableAsHashMapKey<typename InputTensorizor::MCTSKey>;

  { InputTensorizor::eval_key(full_state) } -> std::same_as<typename InputTensorizor::EvalKey>;
  { InputTensorizor::mcts_key(full_state) } -> std::same_as<typename InputTensorizor::MCTSKey>;
  { InputTensorizor::tensorize(&base_state1, &base_state2) } -> std::same_as<typename InputTensorizor::Tensor>;
};

}  // namespace concepts
}  // namespace core
