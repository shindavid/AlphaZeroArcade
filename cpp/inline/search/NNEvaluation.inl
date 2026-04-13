#include "search/NNEvaluation.hpp"

#include "util/MetaProgramming.hpp"

#include <new>

namespace search {

template <core::concepts::Game Game, typename InputFrame, typename NetworkHeads>
void NNEvaluation<Game, InputFrame, NetworkHeads>::init(const InitParams& params) {
  float* data_ptr = init_data_and_offsets(params.valid_moves.size());

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto Index) {
    using Head = mp::TypeAt_t<NetworkHeads, Index>;
    auto& src = std::get<Index>(params.outputs);
    Head::load(data_helper(data_ptr, Index), src, params);
  });

  data_ = data_ptr;
}

template <core::concepts::Game Game, typename InputFrame, typename NetworkHeads>
void NNEvaluation<Game, InputFrame, NetworkHeads>::uniform_init(int num_valid_moves) {
  float* data_ptr = init_data_and_offsets(num_valid_moves);

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto Index) {
    using Head = mp::TypeAt_t<NetworkHeads, Index>;
    Head::uniform_init(data_helper(data_ptr, Index), num_valid_moves);
  });

  data_ = data_ptr;
}

template <core::concepts::Game Game, typename InputFrame, typename NetworkHeads>
bool NNEvaluation<Game, InputFrame, NetworkHeads>::decrement_ref_count() {
  // NOTE: during normal program execution, this is performed in a thread-safe manner. On the
  // other hand, when the program is shutting down, it is not. Thankfully, we don't require thread
  // safety during that phase of the program. If for some reason that changes, we will need to
  // use std::atomic
  ref_count_--;
  return ref_count_ == 0;
}

template <core::concepts::Game Game, typename InputFrame, typename NetworkHeads>
void NNEvaluation<Game, InputFrame, NetworkHeads>::clear() {
  aux_ = nullptr;
  eval_sequence_id_ = 0;
  ref_count_ = 0;

  if (data_) {
    ::operator delete[](data_, std::align_val_t{16});
    data_ = nullptr;
  }
}

template <core::concepts::Game Game, typename InputFrame, typename NetworkHeads>
float* NNEvaluation<Game, InputFrame, NetworkHeads>::init_data_and_offsets(int num_valid_moves) {
  int offset = 0;

  mp::constexpr_for<0, kNumOutputs, 1>([&](auto i) {
    using Head = mp::TypeAt_t<NetworkHeads, i>;
    int size = Head::size(num_valid_moves);

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
