#include "nnet/NNEvaluation.hpp"

#include "util/BitSet.hpp"
#include "util/EigenUtil.hpp"
#include "util/MetaProgramming.hpp"

namespace nnet {

template <core::concepts::EvalSpec EvalSpec>
void NNEvaluation<EvalSpec>::init(OutputTensorTuple& outputs, const ActionMask& valid_actions,
                                  group::element_t sym, core::seat_index_t active_seat,
                                  core::action_mode_t mode) {
  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);

  init_data_and_offsets(valid_actions);

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto Index) {
    using Target = mp::TypeAt_t<PrimaryTargets, Index>;
    using Tensor = Target::Tensor;

    auto& src = std::get<Index>(outputs);
    constexpr bool is_value_based = detail::IsValueBased<Target>::value;
    constexpr bool is_policy_based = detail::IsPolicyBased<Target>::value;
    constexpr bool uses_logit_scale = detail::UsesLogitScale<Target>::value;

    if constexpr (is_value_based) {
      Game::GameResults::right_rotate(src, active_seat);
    }

    using Dst = std::conditional_t<is_policy_based, LocalPolicyTensor, Tensor>;
    using DstMap = Eigen::TensorMap<Dst, Eigen::Aligned>;
    DstMap dst(data(Index), 0);

    if constexpr (is_policy_based) {
      Game::Symmetries::apply(src, inv_sym, mode);

      int i = 0;
      for (core::action_t a : bitset_util::on_indices(valid_actions)) {
        dst(i++) = src(a);
      }
    } else {
      dst = src;
    }

    if constexpr (uses_logit_scale) {
      dst = eigen_util::softmax(dst);
    }
  });
}

template <core::concepts::EvalSpec EvalSpec>
void NNEvaluation<EvalSpec>::uniform_init(const ActionMask& valid_actions) {
  init_data_and_offsets(valid_actions);

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto Index) {
    using Target = mp::TypeAt_t<PrimaryTargets, Index>;
    using Tensor = Target::Tensor;

    constexpr bool is_policy_based = detail::IsPolicyBased<Target>::value;
    using Dst = std::conditional_t<is_policy_based, LocalPolicyTensor, Tensor>;
    using DstMap = Eigen::TensorMap<Dst, Eigen::Aligned>;
    DstMap dst(data(Index), 0);

    Target::uniform_init(valid_actions, dst);
  });
}

template <core::concepts::EvalSpec EvalSpec>
bool NNEvaluation<EvalSpec>::decrement_ref_count() {
  // NOTE: during normal program execution, this is performed in a thread-safe manner. On the
  // other hand, when the program is shutting down, it is not. Thankfully, we don't require thread
  // safety during that phase of the program. If for some reason that changes, we will need to
  // use std::atomic
  ref_count_--;
  return ref_count_ == 0;
}

template <core::concepts::EvalSpec EvalSpec>
void NNEvaluation<EvalSpec>::clear() {
  aux_ = nullptr;
  eval_sequence_id_ = 0;
  ref_count_ = 0;
  delete[] data_;
  data_ = nullptr;
}

template <core::concepts::EvalSpec EvalSpec>
void NNEvaluation<EvalSpec>::init_data_and_offsets(const ActionMask& valid_actions) {
  int offset = 0;

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto i) {
    using Target = mp::TypeAt_t<PrimaryTargets, i>;

    int size = Target::Tensor::Dimensions::total_size;
    if (detail::IsPolicyBased<Target>::value) {
      size = valid_actions.count();
    }

    // pad size so it's a multiple of 4 for alignment (4 * sizeof(float) = 16 bytes)
    int padded_size = (size + 3) & ~3;
    if (i > 0) {
      offsets_[i - 1] = offset;
    }
    offset += padded_size;
  });

  data_ = new float[offset];
}

}  // namespace nnet
