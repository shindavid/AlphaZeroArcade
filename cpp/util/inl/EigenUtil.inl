#include <util/EigenUtil.hpp>

#include <array>
#include <cstdint>

namespace eigen_util {

template<typename Scalar>
template<typename Array, typename Urng, typename... DimTs>
Array UniformDirichletGen<Scalar>::generate(Urng&& urng, Scalar alpha, DimTs&&... dims) {
  static_assert(Array::MaxColsAtCompileTime > 0);
  static_assert(Array::MaxRowsAtCompileTime > 0);

  if (alpha != alpha_) {
    alpha_ = alpha;
    new(&gamma_) GammaGen(alpha);
  }

  Array out(dims...);
  for (int i = 0; i < out.size(); ++i) {
    out.data()[i] = gamma_.template generate<Eigen::Array<Scalar, 1, 1>>(1, 1, urng)(0);
  }
  out /= out.sum();
  return out;
}

template <typename Scalar, int Rows, int Cols, int Options>
auto to_array1d(const Eigen::Array<Scalar, Rows, Cols, Options>& array) {
  static_assert(Rows>0);
  static_assert(Cols>0);
  constexpr int N = Rows * Cols;
  using Array1D = Eigen::Array<Scalar, N, 1>;
  Array1D a;
  for (int i = 0; i < N; ++i) {
    a(i) = array.data()[i];
  }

  return a;
}

template<typename Array> auto softmax(const Array& array) {
  auto normalized_array = array - array.maxCoeff();
  auto z = normalized_array.exp();
  return z / z.sum();
}

template<FixedTensorConcept Tensor> auto reverse(const Tensor& tensor, int dim) {
  using Sizes = extract_sizes_t<Tensor>;
  constexpr int N = Sizes::count;
  Eigen::array<bool, N> rev;
  rev.fill(false);
  rev[dim] = true;
  return tensor.reverse(rev);
}

template<typename Scalar, int Rows, int Options, int MaxRows>
int argmax(const Eigen::Array<Scalar, Rows, 1, Options, MaxRows, 1>& arr) {
  int max_idx = 0;
  Scalar max_val = arr(0);
  for (int i = 1; i < arr.size(); ++i) {
    if (arr(i) > max_val) {
      max_idx = i;
      max_val = arr(i);
    }
  }
  return max_idx;
}

}  // namespace eigen_util
