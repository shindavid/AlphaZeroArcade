#include <util/EigenUtil.hpp>

#include <array>
#include <cstdint>

namespace eigen_util {

template <typename Scalar, int Rows, int Cols, int Options>
auto to_vector(const Eigen::Matrix<Scalar, Rows, Cols, Options>& matrix) {
  static_assert(Rows>0);
  static_assert(Cols>0);
  constexpr int N = Rows * Cols;
  using Vector = Eigen::Vector<Scalar, N>;
  Vector v;

  for (int i = 0; i < N; ++i) {
    v(i) = matrix.data()[i];
  }

  return v;
}

template<typename Vector> auto softmax(const Vector& vector) {
  auto normalized_vector = vector.array() - vector.maxCoeff();
  auto z = normalized_vector.exp();
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

}  // namespace eigen_util
