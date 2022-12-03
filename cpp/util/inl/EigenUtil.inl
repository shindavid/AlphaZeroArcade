#include <util/EigenUtil.hpp>

#include <array>
#include <cstdint>

namespace eigen_util {

template<typename Vector> auto softmax(const Vector& vector) {
  Eigen::Vector<float, 3> v;
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
