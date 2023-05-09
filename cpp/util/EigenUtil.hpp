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

// eigen_util::Shape<...> is a type alias for Eigen::Sizes<...>
template<int64_t... Is> using Shape = Eigen::Sizes<Is...>;

/*
 * ShapeConcept<T> is for concept requirements.
 */
template<typename T> struct is_eigen_shape { static const bool value = false; };
template<int64_t... Is> struct is_eigen_shape<Eigen::Sizes<Is...>> { static const bool value = true; };
template<typename T> inline constexpr bool is_eigen_shape_v = is_eigen_shape<T>::value;
template <typename T> concept ShapeConcept = is_eigen_shape_v<T>;

/*
 * prepend_dim_t<10, Eigen::Sizes<1, 2, 3>> is Eigen::Sizes<10, 1, 2, 3>
 */
template<int64_t N, ShapeConcept Shape> struct prepend_dim {};
template<int64_t N, int64_t... Is> struct prepend_dim<N, Eigen::Sizes<Is...>> {
  using type = Eigen::Sizes<N, Is...>;
};
template<int64_t N, ShapeConcept Shape> using prepend_dim_t = typename prepend_dim<N, Shape>::type;

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

template<FixedTensorConcept FixedTensorT>
struct packed_fixed_tensor_size {
  static constexpr int value = sizeof(typename FixedTensorT::Scalar) * FixedTensorT::Dimensions::total_size;
};
template<FixedTensorConcept FixedTensorT> constexpr int packed_fixed_tensor_size_v = packed_fixed_tensor_size<FixedTensorT>::value;

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
template<FixedTensorConcept DstTensorT, typename SrcTensorT,
    bool Aligned=(packed_fixed_tensor_size_v<DstTensorT> % 8 == 0)>
void packed_fixed_tensor_cp(DstTensorT& dst, const SrcTensorT& src);

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

template<typename T> struct total_size {};
template<ptrdiff_t... Indices> struct total_size<Eigen::Sizes<Indices...>> {
  static constexpr ptrdiff_t size = Eigen::internal::arg_prod(Indices...);
};
template<typename T> constexpr ptrdiff_t total_size_v = total_size<T>::size;

/*
 * Accepts an eigen Tensor and an int row.
 *
 * Reinterprets the tensor as being of shape (N, Shape...) and returns the row-th slice of the tensor, as an
 * Eigen::TensorFixedSize of shape Shape.
 *
 * Beware! Slices are not aligned, which breaks some assumptions made by Eigen. Use at your own risk!
 */
template<ShapeConcept Shape, typename TensorT> const auto& slice(const TensorT& tensor, int row);
template<ShapeConcept Shape, typename TensorT> auto& slice(TensorT& tensor, int row);


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
 * Flattens a fixed-size tensor into an Eigen::Array<Scalar, N, 1>
 */
template<typename Scalar, ShapeConcept Shape, int Options>
const auto& reinterpret_as_array(const Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor);

template<typename Scalar, ShapeConcept Shape, int Options>
auto& reinterpret_as_array(Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor);

/*
 * Inverse of reinterpret_as_array()
 */
template<ShapeConcept Shape, typename Scalar, int N>
const auto& reinterpret_as_tensor(const Eigen::Array<Scalar, N, 1>& array);

template<ShapeConcept Shape, typename Scalar, int N>
auto& reinterpret_as_tensor(Eigen::Array<Scalar, N, 1>& array);

/*
 * sum(), max(), and min() return a 1-element Eigen::TensorFixedSize. To convert to a scalar, access via (0)
 */
template<typename TensorT> auto sum(const TensorT& tensor);
template<typename TensorT> auto max(const TensorT& tensor);
template<typename TensorT> auto min(const TensorT& tensor);


}  // namespace eigen_util

#include <util/inl/EigenUtil.inl>
