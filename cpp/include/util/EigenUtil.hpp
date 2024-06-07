#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include <Eigen/Core>
#include <EigenRand/EigenRand>
#include <boost/mp11.hpp>
#include <torch/torch.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <util/CppUtil.hpp>

/*
 * Various util functions that make the eigen3 library more pleasant to use.
 */
namespace eigen_util {

// eigen_util::Shape<...> is a type alias for Eigen::Sizes<...>
template <int64_t... Is>
using Shape = Eigen::Sizes<Is...>;

/*
 * eigen_util::concepts::Shape<T> is for concept requirements.
 */
template <typename T>
struct is_eigen_shape {
  static const bool value = false;
};
template <int64_t... Is>
struct is_eigen_shape<Eigen::Sizes<Is...>> {
  static const bool value = true;
};
template <typename T>
inline constexpr bool is_eigen_shape_v = is_eigen_shape<T>::value;

namespace concepts {

template <typename T>
concept Shape = is_eigen_shape_v<T>;

}  // namespace concepts

/*
 * rank_v<Eigen::Sizes<...>> is the rank of the Eigen::Sizes.
 */
template <typename T>
struct rank {};
template <int64_t... Is>
struct rank<Eigen::Sizes<Is...>> {
  static constexpr int value = sizeof...(Is);
};
template <typename T>
inline constexpr int rank_v = rank<T>::value;

/*
 * 10 == extract_dim_v<0, Eigen::Sizes<10, 20, 30>>
 * 20 == extract_dim_v<1, Eigen::Sizes<10, 20, 30>>
 * 30 == extract_dim_v<2, Eigen::Sizes<10, 20, 30>>
 */
template <int N, typename T>
struct extract_dim {};

template <int N, int64_t... Is>
struct extract_dim<N, Eigen::Sizes<Is...>> {
  static constexpr int64_t value = util::get_value(std::integer_sequence<int64_t, Is...>{}, N);
};
template <int N, typename T>
constexpr int64_t extract_dim_v = extract_dim<N, T>::value;

/*
 * prepend_dim_t<10, Eigen::Sizes<1, 2, 3>> is Eigen::Sizes<10, 1, 2, 3>
 */
template <int64_t N, concepts::Shape Shape>
struct prepend_dim {};
template <int64_t N, int64_t... Is>
struct prepend_dim<N, Eigen::Sizes<Is...>> {
  using type = Eigen::Sizes<N, Is...>;
};
template <int64_t N, concepts::Shape Shape>
using prepend_dim_t = typename prepend_dim<N, Shape>::type;

/*
 * subshape_t<Eigen::Sizes<10, 20, 30>> is Eigen::Sizes<20, 30>
 *
 * This undoes the effect of prepend_dim_t.
 */
template <concepts::Shape Shape>
struct subshape {};
template <int64_t I, int64_t... Is>
struct subshape<Eigen::Sizes<I, Is...>> {
  using type = Eigen::Sizes<Is...>;
};
template <concepts::Shape Shape>
using subshape_t = typename subshape<Shape>::type;

/*
 * This serves the same role as Eigen::Rand::DirichletGen. However, that implementation is not
 * well-suited for usages with: (1) fixed dimension Matrices, and (2) a uniform alpha distribution.
 *
 * This implementation supports only the uniform-alpha case. When fixed-size matrices are used, it
 * avoids unnecessary dynamic memory allocation.
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
template <typename Scalar>
class UniformDirichletGen {
 public:
  template <typename Array, typename Urng, typename... DimTs>
  Array generate(Urng&& urng, Scalar alpha, DimTs&&... dims);

 private:
  using GammaGen = Eigen::Rand::GammaGen<Scalar>;
  GammaGen gamma_;
  Scalar alpha_ = 1.0;
};

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
template <typename T>
struct to_int64_std_array {};
template <typename I, I... Ints>
struct to_int64_std_array<Eigen::Sizes<Ints...>> {
  static constexpr auto value = std::array<int64_t, sizeof...(Ints)>{Ints...};
};
template <typename T>
auto to_int64_std_array_v = to_int64_std_array<T>::value;

/*
 * The following are equivalent:
 *
 * using T = Eigen::TensorFixedSize<float, Eigen::Sizes<1, 2, 3>, Eigen::RowMajor>;
 *
 * and:
 *
 * using S = Eigen::Sizes<1, 2, 3>;
 * using T = eigen_util::FTensor<S>;
 *
 * The reason we default to RowMajor is for smooth interoperability with pytorch, which is row-major
 * by default.
 *
 * The "f" stands for "fixed-size".
 */
template <concepts::Shape Shape>
using FTensor = Eigen::TensorFixedSize<float, Shape, Eigen::RowMajor>;

template <typename T>
struct is_ftensor {
  static const bool value = false;
};
template <typename Shape>
struct is_ftensor<FTensor<Shape>> {
  static const bool value = true;
};
template <typename T>
inline constexpr bool is_ftensor_v = is_ftensor<T>::value;

namespace concepts {

template <typename T>
concept FTensor = is_ftensor_v<T>;

}  // namespace concepts

// DArray is a dynamic float Eigen::Array of max size N
template <int N>
using DArray = Eigen::Array<float, Eigen::Dynamic, 1, 0, N>;

// FArray is a fixed-size float Eigen::Array of size N
template <int N>
using FArray = Eigen::Array<float, N, 1>;

/*
 * The following are equivalent:
 *
 * using S = Eigen::Sizes<1, 2, 3>;
 *
 * using T = eigen_util::FTensor<Eigen::Sizes<1, 2, 3>>;
 * using S = extract_shape_t<T>;
 */
template <typename T>
struct extract_shape {};
template <concepts::Shape Shape>
struct extract_shape<FTensor<Shape>> {
  using type = Shape;
};
template <typename T>
using extract_shape_t = typename extract_shape<T>::type;

/*
 * serialize() copies the bytes from tensor.data() to buf, checking to make sure it won't overflow
 * past buf_size. Returns the number of bytes written.
 *
 * deserialize() is the inverse operation.
 */
template <concepts::FTensor Tensor>
size_t serialize(char* buf, size_t buf_size, const Tensor& tensor);
template <concepts::FTensor Tensor>
void deserialize(const char* buf, Tensor* tensor);

/*
 * Returns a float array of the same shape as the input, whose values are positive and summing to 1.
 */
template <typename Array>
auto softmax(const Array& arr);

/*
 * Reverses the elements of tensor along the given dimension.
 *
 * This is a convenience wrapper to tensor.reverse(), as tensor.reverse() has a bulkier API.
 *
 * Note that this returns a tensor *operator*, not a tensor.
 */
template <concepts::FTensor Tensor>
auto reverse(const Tensor& tensor, int dim);

/*
 * Accepts a D-dimensional tensor. Randomly samples an index from the tensor, with each index
 * picked proportionally to the value of the tensor at that index.
 *
 * Returns the index as a std::array<int64_t, D>
 */
template <concepts::FTensor Tensor>
auto sample(const Tensor& tensor);

/*
 * Divides tensor by its sum.
 *
 * If the sum is less than eps, then tensor is left unchanged and returns false. Otherwise,
 * returns true.
 */
template <concepts::FTensor Tensor>
bool normalize(Tensor& tensor, double eps = 1e-8);

/*
 * Uniformly randomly picks n nonzero elements of tensor and sets them to zero.
 *
 * Requires that tensor contains at least n nonzero elements.
 */
template <concepts::FTensor Tensor>
void randomly_zero_out(Tensor& tensor, int n);

/*
 * Returns the std::array that fills in the blank in this analogy problem:
 *
 * tensor.data() is to flat_index as tensor is to _____
 */
template <concepts::FTensor Tensor>
auto unflatten_index(const Tensor& tensor, int flat_index);

/*
 * Reinterpret a fixed-size tensor as an Eigen::Array<Scalar, N, 1>
 *
 * auto& array = reinterpret_as_array(tensor);
 */
template <concepts::FTensor FTensor>
const auto& reinterpret_as_array(const FTensor& tensor);

template <concepts::FTensor FTensor>
auto& reinterpret_as_array(FTensor& tensor);

/*
 * Convenience methods that return scalars.
 */
template <typename TensorT>
typename TensorT::Scalar sum(const TensorT& tensor);
template <typename TensorT>
typename TensorT::Scalar max(const TensorT& tensor);
template <typename TensorT>
typename TensorT::Scalar min(const TensorT& tensor);
template <typename TensorT>
bool any(const TensorT& tensor);
template <typename TensorT>
int count(const TensorT& tensor);

/*
 * left_rotate([0, 1, 2, 3], 0) -> [0, 1, 2, 3]
 * left_rotate([0, 1, 2, 3], 1) -> [1, 2, 3, 0]
 * left_rotate([0, 1, 2, 3], 2) -> [2, 3, 0, 1]
 * left_rotate([0, 1, 2, 3], 3) -> [3, 0, 1, 2]
 *
 * right_rotate([0, 1, 2, 3], 0) -> [0, 1, 2, 3]
 * right_rotate([0, 1, 2, 3], 1) -> [3, 0, 1, 2]
 * right_rotate([0, 1, 2, 3], 2) -> [2, 3, 0, 1]
 * right_rotate([0, 1, 2, 3], 3) -> [1, 2, 3, 0]
 */
template <typename Scalar, int N>
void left_rotate(Eigen::Array<Scalar, N, 1>& array, int n);
template <typename Scalar, int N>
void right_rotate(Eigen::Array<Scalar, N, 1>& array, int n);

template <concepts::FTensor TensorT>
uint64_t hash(const TensorT& tensor);

}  // namespace eigen_util

#include <inline/util/EigenUtil.inl>
