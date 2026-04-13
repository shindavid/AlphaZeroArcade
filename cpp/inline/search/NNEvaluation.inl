#include "search/NNEvaluation.hpp"

#include "util/EigenUtil.hpp"
#include "util/MetaProgramming.hpp"

#include <new>

namespace search {

template <::alpha0::concepts::Spec Spec>
void NNEvaluation<Spec>::init(OutputTensorTuple& outputs, const MoveSet& valid_moves,
                                    const InputFrame& frame, group::element_t sym,
                                    core::seat_index_t active_seat) {
  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);

  float* data_ptr = init_data_and_offsets(valid_moves.size());

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto Index) {
    using Head = mp::TypeAt_t<NetworkHeads, Index>;
    using Tensor = Head::Tensor;
    using Shape = Tensor::Dimensions;

    auto& src = std::get<Index>(outputs);

    if constexpr (Head::kGameResultBased) {
      GameResultEncoding::right_rotate(src, active_seat);
    }

    // TODO: do this right_rotate *after* the chip-loop, so we only rotate rows that we actually use
    if constexpr (Head::kWinShareBased) {
      eigen_util::right_rotate(src, active_seat);
    }

    using LocalTensor = Eigen::Tensor<float, eigen_util::extract_rank_v<Shape>, Eigen::RowMajor>;
    using Dst = std::conditional_t<Head::kPerActionBased, LocalTensor, Tensor>;
    using DstMap = Eigen::TensorMap<Dst, Eigen::Aligned>;
    auto arr = eigen_util::to_int64_std_array_v<Shape>;
    if constexpr (Head::kPerActionBased) {
      arr[0] = valid_moves.size();
    }
    DstMap dst(data_helper(data_ptr, Index), arr);

    if constexpr (Head::kPerActionBased) {
      Spec::Symmetries::apply(src, inv_sym, frame);

      int i = 0;
      for (Move move : valid_moves) {
        auto index = PolicyEncoding::to_index(frame, move);
        // We resort to a pragma here to silence an overzealous gcc warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
        dst.chip(i++, 0) = eigen_util::chip_recursive(src, index);
#pragma GCC diagnostic pop
      }
    } else {
      dst = src;
    }

    Head::transform(dst);
  });

  data_ = data_ptr;
}

template <::alpha0::concepts::Spec Spec>
void NNEvaluation<Spec>::uniform_init(int num_valid_moves) {
  float* data_ptr = init_data_and_offsets(num_valid_moves);

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto Index) {
    using Head = mp::TypeAt_t<NetworkHeads, Index>;
    using Tensor = Head::Tensor;
    using Shape = Tensor::Dimensions;

    using LocalTensor = Eigen::Tensor<float, eigen_util::extract_rank_v<Shape>, Eigen::RowMajor>;
    using Dst = std::conditional_t<Head::kPerActionBased, LocalTensor, Tensor>;
    using DstMap = Eigen::TensorMap<Dst, Eigen::Aligned>;

    auto arr = eigen_util::to_int64_std_array_v<Shape>;
    if constexpr (Head::kPerActionBased) {
      arr[0] = num_valid_moves;
    }
    DstMap dst(data_helper(data_ptr, Index), arr);
    Head::uniform_init(dst);
  });

  data_ = data_ptr;
}

template <::alpha0::concepts::Spec Spec>
bool NNEvaluation<Spec>::decrement_ref_count() {
  // NOTE: during normal program execution, this is performed in a thread-safe manner. On the
  // other hand, when the program is shutting down, it is not. Thankfully, we don't require thread
  // safety during that phase of the program. If for some reason that changes, we will need to
  // use std::atomic
  ref_count_--;
  return ref_count_ == 0;
}

template <::alpha0::concepts::Spec Spec>
void NNEvaluation<Spec>::clear() {
  aux_ = nullptr;
  eval_sequence_id_ = 0;
  ref_count_ = 0;

  if (data_) {
    ::operator delete[](data_, std::align_val_t{16});
    data_ = nullptr;
  }
}

template <::alpha0::concepts::Spec Spec>
float* NNEvaluation<Spec>::init_data_and_offsets(int num_valid_moves) {
  int offset = 0;

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto i) {
    using Head = mp::TypeAt_t<NetworkHeads, i>;
    using Tensor = Head::Tensor;
    using Shape = Tensor::Dimensions;

    int size = Head::Tensor::Dimensions::total_size;
    if constexpr (Head::kPerActionBased) {
      size = (size / eigen_util::extract_dim_v<0, Shape>)*num_valid_moves;
    }

    // pad size so it's a multiple of 4 for alignment (4 * sizeof(float) = 16 bytes)
    int padded_size = (size + 3) & ~3;
    if (i > 0) {
      offsets_[i - 1] = offset;
    }
    offset += padded_size;
  });

  RELEASE_ASSERT(!data_);

  // We want to do:
  //
  // return new float[offset];
  //
  // But that doesn't guarantee 16-byte alignment. So we do this instead:
  std::size_t bytes = offset * sizeof(float);
  return static_cast<float*>(::operator new[](bytes, std::align_val_t{16}));
}

}  // namespace search
