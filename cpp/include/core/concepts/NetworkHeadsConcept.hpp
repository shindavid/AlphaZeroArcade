#pragma once

#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/MetaProgramming.hpp"

#include <concepts>

namespace core {

namespace concepts {

template <typename T>
concept NetworkHead = requires(typename T::Tensor& tensor, float* data, int num_valid_moves) {
  // Head name, which must match name used in python.
  { util::decay_copy(T::kName) } -> std::same_as<const char*>;

  typename T::Tensor;
  requires eigen_util::concepts::FTensor<typename T::Tensor>;

  // Returns the logical number of float values that this head stores for the given number of valid
  // moves. NNEvaluation is responsible for padding this for alignment.
  { T::size(num_valid_moves) } -> std::same_as<int>;

  // Uniformly initializes the head data. Used in gen-0 when we don't have an NN.
  { T::uniform_init(data, num_valid_moves) };
};

}  // namespace concepts

template <typename T>
struct _IsNetworkHead {
  static constexpr bool value = concepts::NetworkHead<T>;
};

namespace concepts {

template <typename TT>
concept NetworkHeads = requires {
  typename TT::List;
  requires mp::IsTypeListSatisfying<typename TT::List, _IsNetworkHead>;
};

}  // namespace concepts
}  // namespace core
