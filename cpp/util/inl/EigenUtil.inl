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

template<typename T, typename S> torch::Tensor eigen2torch(fixed_tensor_t<T, S>& tensor) {
  return torch::from_blob(tensor.data(), to_int64_std_array_v<S>);
}

template<typename T, int S> torch::Tensor eigen2torch(Eigen::Vector<T, S>& vector) {
  std::array<int64_t, 1> arr{S};
  return torch::from_blob(vector.data(), arr);
}

}  // namespace eigen_util
