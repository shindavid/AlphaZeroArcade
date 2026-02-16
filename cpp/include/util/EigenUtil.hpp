#pragma once

#include "util/CppUtil.hpp"

#include <Eigen/Core>
#include <EigenRand/EigenRand>
#include <boost/json.hpp>
#include <boost/mp11.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include <array>
#include <cstdint>
#include <map>
#include <type_traits>

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
 * 3 == extract_rank_v<Eigen::Sizes<10, 20, 30>>
 * 1 == extract_rank_v<Eigen::Sizes<5>>
 */
template <typename T>
struct extract_rank {};

template <int64_t... Is>
struct extract_rank<Eigen::Sizes<Is...>> {
  static constexpr int64_t value = sizeof...(Is);
};
template <typename T>
constexpr int64_t extract_rank_v = extract_rank<T>::value;

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
 * This serves the same role as Eigen::Rand::DirichletGen. However, that implementation is not
 * well-suited for usages with: (1) fixed size structures (Array/Matrix), and (2) a uniform alpha
 * distribution.
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
 * The reason we default to RowMajor is for smooth interoperability with TensorRT, which is
 * row-major by default.
 *
 *
 * The "f" stands for "fixed-size".
 */
template <concepts::Shape Shape>
using FTensor = Eigen::TensorFixedSize<float, Shape, Eigen::RowMajor>;

// DArray is a dynamic Eigen::Array of max size N
template <int N, typename Scalar = float>
using DArray = Eigen::Array<Scalar, Eigen::Dynamic, 1, 0, N>;

// FArray is a fixed-size float Eigen::Array of size N
template <int N>
using FArray = Eigen::Array<float, N, 1>;

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

template <typename T>
struct is_farray {
  static const bool value = false;
};
template <int N>
struct is_farray<FArray<N>> {
  static const bool value = true;
};
template <typename T>
inline constexpr bool is_farray_v = is_farray<T>::value;

namespace concepts {

template <typename T>
concept FTensor = is_ftensor_v<T>;

template <typename T>
concept FArray = is_farray_v<T>;

}  // namespace concepts

template <typename T>
struct extract_length {};
template <int N>
struct extract_length<FArray<N>> {
  static constexpr int value = N;
};
template <typename T>
inline constexpr int extract_length_v = extract_length<T>::value;

template <typename Derived>
struct is_eigen_array : std::is_base_of<Eigen::ArrayBase<Derived>, Derived> {};

/*
 * Accepts an Eigen::Array, and sorts the columns based on the values in the first row.
 */
template <typename Derived>
auto sort_columns(const Eigen::ArrayBase<Derived>& array, int row_ix = 0, bool ascending = true);

/*
 * Accepts an Eigen::Array, and sorts the rows based on the values in the first column.
 */
template <typename Derived>
auto sort_rows(const Eigen::ArrayBase<Derived>& array, int col_ix = 0, bool ascending = true);

/*
 * Performs a global-softmax on the input array, in place.
 */
template <class Derived>
void softmax_in_place(Eigen::ArrayBase<Derived>&);

/*
 * Performs a global-softmax on the input tensor, in place.
 */
template <class Derived>
void softmax_in_place(Eigen::TensorBase<Derived, Eigen::WriteAccessors>&);

// Performs a row-wise softmax on the input tensor, in place.
template <class Derived>
void rowwise_softmax_in_place(Eigen::TensorBase<Derived, Eigen::WriteAccessors>&);

/*
 * Applies an element-wise sigmoid function to the input tensor, in place.
 */
template <class Derived>
void sigmoid_in_place(Eigen::TensorBase<Derived, Eigen::WriteAccessors>&);

/*
 * Returns the index of the maximum element. Only works for 1D Array or Matrix.
 */
template <typename Derived>
int argmax(const Eigen::DenseBase<Derived>& arr);

/*
 * Returns a sliced array according to the given indices. The indices are assumed to be of 1D shape.
 * data could be a 1D or 2D array. When slicing a 2D array, it is performed along the first dim (0).
 * e.g. slice([1, 2, 3, 4], [0, 2]) -> [1, 3]
 * e.g. slice([[1, 2], [3, 4], [5, 6]], [0, 2]) -> [[1, 2], [5, 6]]
 */
template <typename Derived1, typename Derived2>
auto slice(const Eigen::ArrayBase<Derived1>& data, const Eigen::ArrayBase<Derived2>& indices);

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
 * Returns a random int k, with probability proportional to T.data()[k].
 */
template <concepts::FTensor Tensor>
int sample(const Tensor& T);

// Like tensor.size(), but works for Eigen::TensorMap<...> as well.
template <class Derived>
int size(const Eigen::TensorBase<Derived>& t);

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
 * Returns what tensor.cwiseMax(x) *should* return. But cwiseMax() doesn't appear to be supported
 * yet.
 */
template <concepts::FTensor Tensor>
auto cwiseMax(const Tensor& tensor, float x);

/*
 * Reinterpret a fixed-size tensor as an Eigen::Array<Scalar, N, 1>
 *
 * auto& array = reinterpret_as_array(tensor);
 */
template <concepts::FTensor Tensor>
const auto& reinterpret_as_array(const Tensor& tensor);

template <concepts::FTensor Tensor>
auto& reinterpret_as_array(Tensor& tensor);

/*
 * Reinterpret a fixed-size array as a rank-1 Eigen::TensorFixedSize<Scalar, Eigen::Sizes<N>>
 */
template <concepts::FArray Array>
const auto& reinterpret_as_tensor(const Array& array);

template <concepts::FArray Array>
auto& reinterpret_as_tensor(Array& array);

// DEBUG_ASSERT()'s that distr is a valid probability distribution
// For release-build's, is a no-op
template <typename T>
void debug_assert_is_valid_prob_distr(const T& distr, float eps = 1e-5);

// Helper functions for debug_assert_is_valid_prob_distr, pulled out for testing.
// Performs check regardless of whether it is a release or debug build.
template <concepts::FTensor Tensor>
void assert_is_valid_prob_distr(const Tensor& distr, float eps = 1e-5);
template <typename Derived>
void assert_is_valid_prob_distr(const Eigen::ArrayBase<Derived>& distr, float eps = 1e-5);

// DEBUG_ASSERT()'s that every value in t is in [min, max]
// For release-build's, is a no-op
template <typename T>
void debug_validate_bounds(const T& t, float min = 0, float max = 1);

template <typename T>
void validate_bounds(const T& t, float min = 0, float max = 1);

/*
 * Convenience methods that return scalars.
 */
template <typename Derived>
float sum(const Eigen::TensorBase<Derived>& tensor);
template <typename Derived>
float max(const Eigen::TensorBase<Derived>& tensor);
template <typename Derived>
float min(const Eigen::TensorBase<Derived>& tensor);
template <concepts::FTensor Tensor>
bool any(const Tensor& tensor);
template <concepts::FTensor Tensor>
bool all(const Tensor& tensor);
template <concepts::FTensor Tensor>
int count(const Tensor& tensor);

template <typename Derived>
bool isfinite(const Eigen::DenseBase<Derived>& x);

template <concepts::FTensor Tensor>
bool isfinite(const Tensor& x);

template <concepts::FTensor Tensor>
bool equal(const Tensor& tensor1, const Tensor& tensor2);

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
template <concepts::FArray Array>
void left_rotate(Array& array, int n);
template <concepts::FArray Array>
void right_rotate(Array& array, int n);

// Rotates the tensor left/right along the *last* dimension.
template <concepts::FTensor Tensor>
void left_rotate(Tensor& tensor, int n);
template <concepts::FTensor Tensor>
void right_rotate(Tensor& tensor, int n);

/*
 * The below functions all accept a tensor of shape (S, ...) as input, and interpret the first
 * Dim**2 <= S elements along the first dimension as entries of a square Dim x Dim board using
 * row-major order. For Dim=8, this looks like this:
 *
 *  0  1  2  3  4  5  6  7
 *  8  9 10 11 12 13 14 15
 * 16 17 18 19 20 21 22 23
 * 24 25 26 27 28 29 30 31
 * 32 33 34 35 36 37 38 39
 * 40 41 42 43 44 45 46 47
 * 48 49 50 51 52 53 54 55
 * 56 57 58 59 60 61 62 63
 *
 * Each function does an in-place transformation of the input tensor.
 *
 * In the function names, "main diagonal" refers to the diagonal from the top-left to the
 * bottom-right, and "anti-diagonal" refers to the diagonal from the top-right to the bottom-left.
 *
 * Implementations are based on:
 *
 * https://stackoverflow.com/a/8664879/543913
 *
 * The awkwardness with the Dim template parameter is because in some games like go and othello,
 * the policy is based on the board positions plus an extra "pass" action. So for a 19x19 board,
 * the policy tensor has shape (362, ...), and we still want to be able to perform these
 * transformations on the first 361 elements.
 */
template <int Dim, concepts::FTensor Tensor>
void rot90_clockwise(Tensor& tensor);
template <int Dim, concepts::FTensor Tensor>
void rot180(Tensor& tensor);
template <int Dim, concepts::FTensor Tensor>
void rot270_clockwise(Tensor& tensor);
template <int Dim, concepts::FTensor Tensor>
void flip_vertical(Tensor& tensor);
template <int Dim, concepts::FTensor Tensor>
void mirror_horizontal(Tensor& tensor);
template <int Dim, concepts::FTensor Tensor>
void flip_main_diag(Tensor& tensor);
template <int Dim, concepts::FTensor Tensor>
void flip_anti_diag(Tensor& tensor);

template <concepts::FTensor Tensor>
uint64_t hash(const Tensor& tensor);

/*
 * Note that compute_covariance() returns a tensor *operator*, not a tensor.
 * This means that a construct like this will almost certainly result in unexpected behavior:
 *
 *   x = compute_covariance(x);
 *
 * See: https://eigen.tuxfamily.org/dox/group__TopicAliasing.html
 */
template <typename Derived>
auto compute_covariance(const Eigen::MatrixBase<Derived>& X);

using PrintArrayFormatMap = std::map<std::string, std::function<std::string(float)>>;

/*
 * Prints a 2D array of size (n_rows, n_cols) to the given output stream.
 *
 * Expects additionally a vector column_names, of size n_cols, that are used as column headers.
 *
 * If fmt_map is not nullptr, it should be a map from column names to functions that convert floats
 * to strings. If fmt_map is nullptr, or if a given column name is not found in fmt_map, then a
 * default string conversion is used.
 */
template <typename Derived>
void print_array(std::ostream& os, const Eigen::ArrayBase<Derived>& array,
                 const std::vector<std::string>& column_names,
                 const PrintArrayFormatMap* fmt_map = nullptr);

template <typename Derived>
boost::json::object output_to_json(const Eigen::ArrayBase<Derived>& array,
                                   const std::vector<std::string>& column_names,
                                   const PrintArrayFormatMap* fmt_map = nullptr);

template <typename Derived0, typename... Deriveds>
auto concatenate_columns(const Eigen::ArrayBase<Derived0>& first,
                         const Eigen::ArrayBase<Deriveds>&... rest);

template <eigen_util::concepts::FTensor T>
boost::json::array to_json(const T& tensor);
template <eigen_util::concepts::FArray T>
boost::json::array to_json(const T& array);

template <concepts::FTensor Tensor>
Tensor zeros();

}  // namespace eigen_util

#include "inline/util/EigenUtil.inl"
