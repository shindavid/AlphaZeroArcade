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
template<typename T> struct to_sizes { using type = T; };
template<int... Ints> struct to_sizes<util::int_sequence<Ints...>> { using type = Eigen::Sizes<Ints...>; };
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
template<typename I, I... Ints> struct to_int64_std_array<Eigen::Sizes<Ints...>> {
  static constexpr auto value = std::array<int64_t, sizeof...(Ints)>{Ints...};
};
template<typename T> auto to_int64_std_array_v = to_int64_std_array<T>::value;

/*
 * The following are equivalent:
 *
 * using T = Eigen::TensorFixedSize<float, Eigen::Sizes<1, 2, 3>, Eigen::RowMajor>;
 *
 * and:
 *
 * using S = Eigen::Sizes<1, 2, 3>;
 * using T = eigen_util::fixed_tensor_t<float, S>;
 *
 * The reason we default to RowMajor is for smooth interoperability with pytorch, which is row-major by default.
 */
template<typename T, typename S> using fixed_tensor_t = Eigen::TensorFixedSize<T, S, Eigen::RowMajor>;

/*
 * FixedTensorConcept is a concept corresponding to fixed_tensor_t
 */
template<typename T> struct is_fixed_tensor { static const bool value = false; };
template<typename T, typename S> struct is_fixed_tensor<fixed_tensor_t<T, S>> { static const bool value = true; };
template<typename T> inline constexpr bool is_fixed_tensor_v = is_fixed_tensor<T>::value;
template<typename T> concept FixedTensorConcept = is_fixed_tensor_v<T>;

/*
 * The following are equivalent:
 *
 * using S = Eigen::Sizes<1, 2, 3>;
 *
 * using T = eigen_util::fixed_tensor_t<float, Eigen::Sizes<1, 2, 3>>;
 * using S = extract_sizes_t<T>;
 */
template<typename T> struct extract_sizes {};
template<typename T, typename S> struct extract_sizes<fixed_tensor_t<T, S>> {
  using type = S;
};
template<typename T> using extract_sizes_t = typename extract_sizes<T>::type;

/*
 * In numpy, you can easily broadcast arrays with scalars, like:
 *
 * a = np.array([1, 2, 3])
 * b = a / sum(a)  # broadcast
 * c = a - max(a)  # broadcast
 *
 * In Eigen, we don't get this nice sort of broadcasting for free:
 *
 * auto b = a / a.maximum();  // returns garbage!
 * auto c = a - a.sum();  // returns garbage!
 *
 * You could try to do the appropriate broadcasting yourself, but this is stupidly tricky, due the presence of a
 * zero-dimensional tensor bug in Eigen 3.4.0: https://stackoverflow.com/a/74158117/543913
 *
 * eigen_util::fixed_op_to_scalar() exists to better simulate the numpy idioms when working with fixed_tensor_t objects.
 * You can use it like so:
 *
 * auto b = a / eigen_util::fixed_op_to_scalar<float>(a.maximum());
 * auto c = a - eigen_util::fixed_op_to_scalar<float>(a.sum());
 */
template<typename T, typename TensorOp> T fixed_op_to_scalar(const TensorOp& op);

/*
 * Returns a float tensor op of the same shape as the input, whose values are positive and summing to 1.
 */
template<FixedTensorConcept Tensor> auto softmax(const Tensor& tensor);

/*
 * Reinterprets an eigen_util::fixed_tensor_t as a torch::Tensor.
 *
 * This is NOT a copy. Modifying the outputted value will result in modifications to the inputted value.
 */
template<typename T, typename S> torch::Tensor eigen2torch(fixed_tensor_t<T, S>& tensor);

}  // namespace eigen_util

#include <util/inl/EigenUtil.inl>
