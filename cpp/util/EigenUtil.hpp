#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include <EigenRand/EigenRand>
#include <boost/mp11.hpp>
#include <torch/torch.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <util/CppUtil.hpp>

namespace eigen_util {

/*
 * This serves the same role as Eigen::Rand::DirichletGen. However, that implementation is not well-suited for
 * usages with: (1) fixed dimension Matrices, and (2) a uniform alpha distribution.
 *
 * This implementation supports only the uniform-alpha case. When fixed-size matrices are used, it avoids
 * unnecessary dynamic memory allocation.
 *
 * Usage:
 *
 * float alpha = 0.03;
 * using Gen = eigen_util::UniformDirichletGen<float>;
 * Gen gen;  // good to reuse same object repeatedly if same alpha will be used repeatedly
 * Eigen::Rand::P8_mt19937_64 rng{ 42 };
 *
 * // fixed size case
 * using Array = Eigen::Array<float, 4, 1>;
 * Array arr = gen.generate<Array>(rng, alpha);
 *
 * // dynamic size case with runtime size
 * using Array = Eigen::Array<float>;
 * Array arr = gen.generate<Array>(rng, alpha, 4, 1);  // returns 4x1 dynamic matrix
 */
template<typename Scalar>
class UniformDirichletGen {
public:
  template<typename Array, typename Urng, typename... DimTs>
  Array generate(Urng&& urng, Scalar alpha, DimTs&&... dims);

private:
  using GammaGen = Eigen::Rand::GammaGen<Scalar>;
  GammaGen gamma_;
  Scalar alpha_ = 1.0;
};

/*
 * Flattens an Array<T, M, N, ..> into an Array<T, M*N, 1>
 */
template <typename Scalar, int Rows, int Cols, int Options>
auto to_array1d(const Eigen::Array<Scalar, Rows, Cols, Options>& array);

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
 * Returns a float array of the same shape as the input, whose values are positive and summing to 1.
 */
template<typename Array> auto softmax(const Array& arr);

/*
 * Reverses the elements of tensor along the given dimension.
 *
 * This is a convenience wrapper to tensor.reverse(), as tensor.reverse() has a bulkier API.
 */
template<FixedTensorConcept Tensor> auto reverse(const Tensor& tensor, int dim);

}  // namespace eigen_util

#include <util/inl/EigenUtil.inl>
