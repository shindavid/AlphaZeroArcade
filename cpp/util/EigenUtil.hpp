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
 * FixedVectorConcept is a concept corresponding to Eigen::Vector<T, N>
 */
template<typename T> struct is_fixed_vector { static const bool value = false; };
template<typename T, int S> struct is_fixed_vector<Eigen::Vector<T, S>> { static const bool value = true; };
template<typename T> inline constexpr bool is_fixed_vector_v = is_fixed_vector<T>::value;
template<typename T> concept FixedVectorConcept = is_fixed_vector_v<T>;

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
 * Returns a float vector op of the same shape as the input, whose values are positive and summing to 1.
 */
template<typename Vector> auto softmax(const Vector& vector);

/*
 * Reinterprets an eigen_util::fixed_tensor_t as a torch::Tensor.
 *
 * This is NOT a copy. Modifying the outputted value will result in modifications to the inputted value.
 */
template<typename T, typename N> torch::Tensor eigen2torch(fixed_tensor_t<T, N>& tensor);

/*
 * Reinterprets an Eigen::Vector as a torch::Tensor of the provided Shape
 *
 * This is NOT a copy. Modifying the outputted value will result in modifications to the inputted value.
 */
template<util::IntSequenceConcept Shape, typename T, int N> torch::Tensor eigen2torch(Eigen::Vector<T, N>& vector);

/*
 * Reverses the elements of tensor along the given dimension.
 *
 * This is a convenience wrapper to tensor.reverse(), as tensor.reverse() has a bulkier API.
 */
template<FixedTensorConcept Tensor> auto reverse(const Tensor& tensor, int dim);

}  // namespace eigen_util

#include <util/inl/EigenUtil.inl>
