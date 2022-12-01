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

template<typename T, typename N> torch::Tensor eigen2torch(fixed_tensor_t<T, N>& tensor) {
  return torch::from_blob(tensor.data(), to_int64_std_array_v<N>);
}

template<util::IntSequenceConcept Shape, typename T, int N> torch::Tensor eigen2torch(Eigen::Vector<T, N>& vector) {
  static_assert(util::int_sequence_product_v<Shape> == N);
  return torch::from_blob(vector.data(), util::std_array_v<int64_t, Shape>);
}

}  // namespace eigen_util
