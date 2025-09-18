#include "search/NNEvaluation.hpp"

#include "util/BitSet.hpp"
#include "util/EigenUtil.hpp"
#include "util/MetaProgramming.hpp"

#include <new>

namespace search {

template <search::concepts::Traits Traits>
void NNEvaluation<Traits>::init(OutputTensorTuple& outputs, const ActionMask& valid_actions,
                                group::element_t sym, core::seat_index_t active_seat,
                                core::action_mode_t mode) {
  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);

  init_data_and_offsets(valid_actions);

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto Index) {
    using Target = mp::TypeAt_t<PrimaryTargets, Index>;
    using Tensor = Target::Tensor;
    using Shape = Tensor::Dimensions;

    auto& src = std::get<Index>(outputs);

    if constexpr (Target::kValueBased) {
      Game::GameResults::right_rotate(src, active_seat);
    }

    using Dst = std::conditional_t<Target::kPolicyBased, LocalPolicyTensor, Tensor>;
    using DstMap = Eigen::TensorMap<Dst, Eigen::Aligned>;
    auto arr = eigen_util::to_int64_std_array_v<Shape>;
    if constexpr (Target::kPolicyBased) {
      arr[0] = valid_actions.count();
    }
    DstMap dst(data(Index), arr);

    if constexpr (Target::kPolicyBased) {
      Game::Symmetries::apply(src, inv_sym, mode);

      int i = 0;
      for (core::action_t a : bitset_util::on_indices(valid_actions)) {
        dst(i++) = src(a);
      }
    } else {
      dst = src;
    }

    Target::transform(dst);
  });
}

template <search::concepts::Traits Traits>
void NNEvaluation<Traits>::uniform_init(const ActionMask& valid_actions) {
  init_data_and_offsets(valid_actions);

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto Index) {
    using Target = mp::TypeAt_t<PrimaryTargets, Index>;
    using Tensor = Target::Tensor;
    using Shape = Tensor::Dimensions;

    using Dst = std::conditional_t<Target::kPolicyBased, LocalPolicyTensor, Tensor>;
    using DstMap = Eigen::TensorMap<Dst, Eigen::Aligned>;
    auto arr = eigen_util::to_int64_std_array_v<Shape>;
    if constexpr (Target::kPolicyBased) {
      arr[0] = valid_actions.count();
    }
    DstMap dst(data(Index), arr);
    Target::uniform_init(dst);
  });
}

template <search::concepts::Traits Traits>
bool NNEvaluation<Traits>::decrement_ref_count() {
  // NOTE: during normal program execution, this is performed in a thread-safe manner. On the
  // other hand, when the program is shutting down, it is not. Thankfully, we don't require thread
  // safety during that phase of the program. If for some reason that changes, we will need to
  // use std::atomic
  ref_count_--;
  return ref_count_ == 0;
}

template <search::concepts::Traits Traits>
void NNEvaluation<Traits>::clear() {
  aux_ = nullptr;
  eval_sequence_id_ = 0;
  ref_count_ = 0;

  if (data_) {
    ::operator delete[](data_, std::align_val_t{16});
    data_ = nullptr;
  }
}

template <search::concepts::Traits Traits>
void NNEvaluation<Traits>::init_data_and_offsets(const ActionMask& valid_actions) {
  int offset = 0;

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto i) {
    using Target = mp::TypeAt_t<PrimaryTargets, i>;

    int size = Target::Tensor::Dimensions::total_size;
    if (Target::kPolicyBased) {
      size = valid_actions.count();
    }

    // pad size so it's a multiple of 4 for alignment (4 * sizeof(float) = 16 bytes)
    int padded_size = (size + 3) & ~3;
    if (i > 0) {
      offsets_[i - 1] = offset;
    }
    offset += padded_size;
  });

  if (data_) {
    ::operator delete[](data_, std::align_val_t{16});
  }

  // We want to do:
  //
  // data_ = new float[offset];
  //
  // But that doesn't guarantee 16-byte alignment. So we do this instead:
  std::size_t bytes = offset * sizeof(float);
  data_ = static_cast<float*>(::operator new[](bytes, std::align_val_t{16}));
}

}  // namespace search
