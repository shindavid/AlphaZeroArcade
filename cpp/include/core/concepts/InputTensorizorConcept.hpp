#pragma once

#include "core/BasicTypes.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <concepts>

namespace core::concepts {

template <typename IT, typename InputFrame>
concept InputTensorizor =
  requires(IT& instance, group::element_t sym, const InputFrame& frame,
           const InputFrame& next_frame, typename IT::StateIterator it, core::action_t action) {
    typename IT::Tensor;
    typename IT::EvalKey;

    requires util::concepts::UsableAsHashMapKey<typename IT::EvalKey>;
    requires eigen_util::concepts::FTensor<typename IT::Tensor>;

    // kNumStatesToEncode is the number of State's that are needed to tensorize a given state. If
    // the neural network does not need any previous State's, kNumStatesToEncode should be 1.
    { util::decay_copy(IT::kNumStatesToEncode) } -> std::same_as<int>;

    { instance.tensorize(sym) } -> std::same_as<typename IT::Tensor>;
    { instance.get_random_symmetry() } -> std::same_as<group::element_t>;
    { instance.get_random_symmetry(next_frame) } -> std::same_as<group::element_t>;
    { instance.undo() } -> std::same_as<void>;
    { instance.jump_to(it) } -> std::same_as<void>;
    { instance.clear() } -> std::same_as<void>;
    { instance.update(frame) } -> std::same_as<void>;
    { instance.current_frame() } -> std::same_as<const InputFrame&>;
    { instance.eval_key() } -> std::same_as<typename IT::EvalKey>;
  };

}  // namespace core::concepts
