#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include <boost/mp11.hpp>
#include <torch/torch.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <util/CppUtil.hpp>

namespace eigen_util {

/*
 * The following are equivalent:
 *
 * using T = Eigen::Sizes<1, 2, 3>;
 *
 * and:
 *
 * using S = util::int_sequence<1, 2, 3>;
 * using T = eigen_util::to_sizes_t<S>;
 *
 * and:
 *
 * using S = Eigen::Sizes<1, 2, 3>;
 * using T = eigen_util::to_sizes_t<S>;
 */
template<typename T> struct to_sizes {
  using type = T;
};
template<int... Ints>
struct to_sizes<util::int_sequence<Ints...>> {
  using type = Eigen::Sizes<Ints...>;
};
template<typename T> using to_sizes_t = typename to_sizes<T>::type;

/*
 * The following are equivalent:
 *
 * using T = std::array<int64_t, 2>(5, 6);
 *
 * and:
 *
 * using S = Eigen::Sizes<5, 6>;
 * using T = to_int64_std_array_v<S>;
 */
template<typename T> struct to_int64_std_array {};
template<int... Ints> struct to_int64_std_array<Eigen::Sizes<Ints...>> {
  static constexpr auto value = std::array<int64_t, sizeof...(Ints)>(Ints...);
};
template<typename T> auto to_int64_std_array_v = to_int64_std_array<T>::value;

/*
 * The following are equivalent:
 *
 * using T = Eigen::TensorFixedSize<float, Eigen::Sizes<1, 2, 3>, Eigen::RowMajor>;
 *
 * and:
 *
 * using S = util::int_sequence<1, 2, 3>;
 * using T = eigen_util::fixed_tensor_t<float, S>;
 *
 * and:
 *
 * using S = Eigen::Sizes<1, 2, 3>;
 * using T = eigen_util::fixed_tensor_t<float, S>;
 *
 * The reason we default to RowMajor is for smooth interoperability with
 * pytorch, which is row-major by default.
 */
template<typename T, typename S>
using fixed_tensor_t = Eigen::TensorFixedSize<T, to_sizes_t<S>, Eigen::RowMajor>;

/*
 * Reinterprets an eigen_util::fixed_tensor_t as a torch::Tensor.
 *
 * This is NOT a copy. Modifying the outputted value will result in
 * modifications to the inputted value.
 */
template<typename T, typename S>
torch::Tensor eigen2torch(const fixed_tensor_t<T, S>& tensor);

}  // namespace eigen_util

#include <util/inl/EigenUtil.inl>
