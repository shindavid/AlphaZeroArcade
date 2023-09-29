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
template<int64_t... Is> using Shape = Eigen::Sizes<Is...>;

/*
 * ShapeConcept<T> is for concept requirements.
 */
template<typename T> struct is_eigen_shape { static const bool value = false; };
template<int64_t... Is> struct is_eigen_shape<Eigen::Sizes<Is...>> { static const bool value = true; };
template<typename T> inline constexpr bool is_eigen_shape_v = is_eigen_shape<T>::value;
template <typename T> concept ShapeConcept = is_eigen_shape_v<T>;

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
template <int64_t N, ShapeConcept Shape>
struct prepend_dim {};
template<int64_t N, int64_t... Is> struct prepend_dim<N, Eigen::Sizes<Is...>> {
  using type = Eigen::Sizes<N, Is...>;
};
template<int64_t N, ShapeConcept Shape> using prepend_dim_t = typename prepend_dim<N, Shape>::type;

/*
 * subshape_t<Eigen::Sizes<10, 20, 30>> is Eigen::Sizes<20, 30>
 *
 * This undoes the effect of prepend_dim_t.
 */
template<ShapeConcept Shape> struct subshape {};
template<int64_t I, int64_t... Is> struct subshape<Eigen::Sizes<I, Is...>> {
  using type = Eigen::Sizes<Is...>;
};
template<ShapeConcept Shape> using subshape_t = typename subshape<Shape>::type;

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
template<typename Scalar, ShapeConcept Shape>
using fixed_tensor_t = Eigen::TensorFixedSize<Scalar, Shape, Eigen::RowMajor>;

/*
 * FixedTensorConcept is a concept corresponding to fixed_tensor_t
 */
template<typename T> struct is_fixed_tensor { static const bool value = false; };
template<typename Scalar, typename Shape> struct is_fixed_tensor<fixed_tensor_t<Scalar, Shape>> {
  static const bool value = true;
};
template<typename T> inline constexpr bool is_fixed_tensor_v = is_fixed_tensor<T>::value;
template<typename T> concept FixedTensorConcept = is_fixed_tensor_v<T>;

template<typename T> struct is_fixed_matrix { static const bool value = false; };
template<typename Scalar, int Rows, int Cols, int Options>
struct is_fixed_matrix<Eigen::Matrix<Scalar, Rows, Cols, Options>> {
  static constexpr bool value = Rows > 0 && Cols > 0;
};
template<typename T> inline constexpr bool is_fixed_matrix_v = is_fixed_matrix<T>::value;
template<typename T> concept FixedMatrixConcept = is_fixed_matrix_v<T>;

// /*
//  * ShapeIndexConcept<T, Shape> is a concept that means that T is a 1D-tensor of dtype int and
//  * of length Shape::count. This means that it can be used to index to a position in a tensor of
//  * shape Shape.
//  */
// template<typename T, typename U> struct is_shape_index { static const bool value = false; };
// template<ShapeConcept Shape1, ShapeConcept Shape2>
// struct is_shape_index<fixed_tensor_t<int, Shape1>, Shape2> {
//   static const bool value = std::is_same_v<Shape1, fixed_tensor_t<int, Shape<Shape2::count>>;
// };
// template<typename T, typename U> inline constexpr bool is_shape_index_v = is_shape_index<T, U>::value;
// template <typename T, typename U> concept ShapeIndexConcept = is_shape_index_v<T, U>;

/*
 * ShapeMaskConcept<T, Shape> is a concept that means that T is a tensor of dtype bool and
 * of shape Shape. This means that it can be used to represent a mask of shape Shape.
 */
template<typename T, typename U> struct is_shape_mask { static const bool value = false; };
template<ShapeConcept Shape>
struct is_shape_mask<fixed_tensor_t<bool, Shape>, Shape> {
  static const bool value = true;
};
template<typename T, typename U> inline constexpr bool is_shape_mask_v = is_shape_mask<T, U>::value;
template <typename T, typename U> concept ShapeMaskConcept = is_shape_mask_v<T, U>;


template<FixedTensorConcept FixedTensorT>
struct packed_fixed_tensor_size {
  static constexpr int value = sizeof(typename FixedTensorT::Scalar) * FixedTensorT::Dimensions::total_size;
};
template<FixedTensorConcept FixedTensorT> constexpr int packed_fixed_tensor_size_v = packed_fixed_tensor_size<FixedTensorT>::value;

template<FixedTensorConcept FixedTensorT>
struct alignment_safe {
  static constexpr bool value = packed_fixed_tensor_size_v<FixedTensorT> % 8 == 0;
};
template<FixedTensorConcept FixedTensorT> constexpr bool alignment_safe_v = alignment_safe<FixedTensorT>::value;


/*
 * The naive way to copy to a Eigen::TensorFixedSize is this:
 *
 * dst = src;
 *
 * However, this results in a segfault if dst is not aligned. See here for details:
 * https://eigen.tuxfamily.org/dox/group__TopicUnalignedArrayAssert.html
 *
 * The above documentation implies that using the Eigen::DontAlign flag when constructing the destination tensor type
 * should fix the problem. However, empirically this does not seem to work.
 *
 * packed_fixed_tensor_cp() is a workaround that avoids the segfault. Whether the destination is aligned or not is
 * by default determined by the size of the destination tensor, but can be overridden by the Aligned template parameter.
 * If Aligned is true, the function simply performs the naive assignment. Otherwise, it uses memcpy().
 *
 * The usage to replace the above line is:
 *
 * packed_fixed_tensor_cp(dst, src);
 *
 * Note that the standard warning about aliasing in Eigen applies: in some cases you should call:
 *
 * packed_fixed_tensor_cp(dst, src.eval());
 *
 * in order to avoid potential aliasing. See: https://eigen.tuxfamily.org/dox/group__TopicAliasing.html
 */
template<FixedTensorConcept DstTensorT, typename SrcTensorT, bool Aligned=alignment_safe_v<DstTensorT>>
void packed_fixed_tensor_cp(DstTensorT& dst, const SrcTensorT& src);

/*
 * The following are equivalent:
 *
 * using S = Eigen::Sizes<1, 2, 3>;
 *
 * using T = eigen_util::fixed_tensor_t<float, Eigen::Sizes<1, 2, 3>>;
 * using S = extract_shape_t<T>;
 */
template<typename T> struct extract_shape {};
template<typename Scalar, ShapeConcept Shape> struct extract_shape<fixed_tensor_t<Scalar, Shape>> {
  using type = Shape;
};
template<typename T> using extract_shape_t = typename extract_shape<T>::type;

/*
 * Accepts an eigen Tensor and an int row.
 *
 * Reinterprets the tensor as being of shape (N, Shape...) and returns the row-th slice of the tensor, as an
 * Eigen::TensorFixedSize of shape Shape.
 *
 * If the input tensor is a TensorFixedSize, then the shape is inferred from the input tensor's shape. Otherwise, the
 * shape needs to be passed as a template parameter.
 *
 * Beware! Slices are not aligned, which breaks some assumptions made by Eigen. Use at your own risk!
 */
template<FixedTensorConcept TensorT> const auto& slice(const TensorT& tensor, int row);
template<FixedTensorConcept TensorT> auto& slice(TensorT& tensor, int row);
template<ShapeConcept Shape, typename TensorT> const auto& slice(const TensorT& tensor, int row);
template<ShapeConcept Shape, typename TensorT> auto& slice(TensorT& tensor, int row);

/*
 * serialize() copies the bytes from tensor.data() to buf, checking to make sure it won't overflow
 * past buf_size. Returns the number of bytes written.
 *
 * deserialize() is the inverse operation.
 */
template<FixedTensorConcept Tensor> size_t serialize(char* buf, size_t buf_size, const Tensor& tensor);
template<FixedTensorConcept Tensor> void deserialize(const char* buf, Tensor* tensor);

/*
 * Returns a float array of the same shape as the input, whose values are positive and summing to 1.
 */
template<typename Array> auto softmax(const Array& arr);

/*
 * Reverses the elements of tensor along the given dimension.
 *
 * This is a convenience wrapper to tensor.reverse(), as tensor.reverse() has a bulkier API.
 *
 * Note that this returns a tensor *operator*, not a tensor.
 */
template<FixedTensorConcept Tensor> auto reverse(const Tensor& tensor, int dim);

/*
 * Flattens a bool tensor and returns a std::bitset of the same size. This is useful for shrinking the memory
 * footprint of a bool tensor, which is 8x larger than a bitset.
 */
template<ShapeConcept Shape>
auto fixed_bool_tensor_to_std_bitset(const Eigen::TensorFixedSize<bool, Shape, Eigen::RowMajor>& tensor);

/*
 * Inverse of fixed_bool_tensor_to_std_bitset().
 */
template<ShapeConcept Shape, size_t N>
auto std_bitset_to_fixed_bool_tensor(const std::bitset<N>& bitset);

/*
 * Accepts a D-dimensional tensor. Randomly samples an index from the tensor, with each index
 * picked proportionally to the value of the tensor at that index.
 *
 * Returns the index as a std::array<int64_t, D>
 */
template<FixedTensorConcept Tensor> auto sample(const Tensor& tensor);

/*
 * Returns the std::array that fills in the blank in this analogy problem:
 *
 * tensor.data() is to flat_index as tensor is to _____
 */
template<FixedTensorConcept Tensor> auto unflatten_index(const Tensor& tensor, int flat_index);

// template<typename Scalar, ShapeConcept Shape, int Options>
// auto from_1d_tensor_to_std_array(const Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor);

/*
 * Reinterpret a fixed-size tensor as an Eigen::Array<Scalar, N, 1>
 *
 * auto& array = reinterpret_as_array(tensor);
 */
template<typename Scalar, ShapeConcept Shape, int Options>
const auto& reinterpret_as_array(const Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor);

template<typename Scalar, ShapeConcept Shape, int Options>
auto& reinterpret_as_array(Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor);

/*
 * Inverse of reinterpret_as_array()
 *
 * auto& tensor = reinterpret_as_tensor<TensorT>(array);
 *
 * or using c++
 */
template<FixedTensorConcept TensorT, typename Scalar, int N>
const TensorT& reinterpret_as_tensor(const Eigen::Array<Scalar, N, 1>& array);

template<FixedTensorConcept TensorT, typename Scalar, int N>
TensorT& reinterpret_as_tensor(Eigen::Array<Scalar, N, 1>& array);

/*
 * Reinterpret a fixed-size tensor as an Eigen::Matrix.
 *
 * auto& tensor = reinterpret_as_matrix<MatrixT>(tensor);
 */
template<FixedMatrixConcept MatrixT, typename Scalar, ShapeConcept Shape, int Options>
const MatrixT& reinterpret_as_matrix(const Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor);

template<FixedMatrixConcept MatrixT, typename Scalar, ShapeConcept Shape, int Options>
MatrixT& reinterpret_as_matrix(Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor);


/*
 * Convenience methods that return scalars.
 */
template<typename TensorT> typename TensorT::Scalar sum(const TensorT& tensor);
template<typename TensorT> typename TensorT::Scalar max(const TensorT& tensor);
template<typename TensorT> typename TensorT::Scalar min(const TensorT& tensor);
template<typename TensorT> bool any(const TensorT& tensor);
template<typename TensorT> int count(const TensorT& tensor);

/*
 * Multiplies the positive elements of array by s.
 */
template<typename Scalar, int N> void positive_scale(Eigen::Array<Scalar, N, 1>& array, Scalar s);


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
template<typename Scalar, int N> void left_rotate(Eigen::Array<Scalar, N, 1>& array, int n);
template<typename Scalar, int N> void right_rotate(Eigen::Array<Scalar, N, 1>& array, int n);

template <FixedTensorConcept TensorT> uint64_t hash(const TensorT& tensor);

}  // namespace eigen_util

#include <util/inl/EigenUtil.inl>
