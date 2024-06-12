#include <util/EigenUtil.hpp>

#include <util/Exception.hpp>
#include <util/Random.hpp>

#include <array>
#include <cstdint>

namespace eigen_util {

template <typename Scalar>
template <typename Array, typename Urng, typename... DimTs>
Array UniformDirichletGen<Scalar>::generate(Urng&& urng, Scalar alpha, DimTs&&... dims) {
  static_assert(Array::MaxColsAtCompileTime > 0);
  static_assert(Array::MaxRowsAtCompileTime > 0);

  if (alpha != alpha_) {
    alpha_ = alpha;
    new (&gamma_) GammaGen(alpha);
  }

  Array out(dims...);
  for (int i = 0; i < out.size(); ++i) {
    out.data()[i] = gamma_.template generate<Eigen::Array<Scalar, 1, 1>>(1, 1, urng)(0);
  }
  out /= out.sum();
  return out;
}

template <typename Array>
auto softmax(const Array& array) {
  auto normalized_array = array - array.maxCoeff();
  auto z = normalized_array.exp();
  return z / z.sum();
}

template <concepts::FTensor Tensor>
auto reverse(const Tensor& tensor, int dim) {
  using Sizes = extract_shape_t<Tensor>;
  constexpr int N = Sizes::count;
  static_assert(N > 0);

  Eigen::array<bool, N> rev;
  rev.fill(false);
  rev[dim] = true;
  return tensor.reverse(rev);
}

template <concepts::FTensor Tensor>
auto sample(const Tensor& tensor) {
  using Shape = extract_shape_t<Tensor>;
  constexpr size_t N = Shape::total_size;

  const auto* data = tensor.data();
  int flat_index = util::Random::weighted_sample(data, data + N);
  return unflatten_index(tensor, flat_index);
}

template <concepts::FTensor Tensor>
bool normalize(Tensor& tensor, double eps) {
  auto s = sum(tensor);
  if (s < eps) return false;

  tensor = tensor / s;
  return true;
}

template <concepts::FTensor Tensor>
void randomly_zero_out(Tensor& tensor, int n) {
  using Shape = extract_shape_t<Tensor>;
  constexpr size_t N = Shape::total_size;

  auto* data = tensor.data();
  util::Random::zero_out(data, data + N, n);
}

template <concepts::FTensor Tensor>
auto unflatten_index(const Tensor& tensor, int flat_index) {
  // Convert the 1D index back to a K-dimensional index
  using Shape = extract_shape_t<Tensor>;
  constexpr size_t K = Shape::count;

  std::array<int64_t, K> index;
  int residual = flat_index;
  for (size_t k = 0; k < K - 1; ++k) {
    index[k] = residual % tensor.dimension(k);
    residual /= tensor.dimension(k);
  }
  index[K - 1] = residual;

  return index;
}

template <concepts::FTensor Tensor>
const auto& reinterpret_as_array(const Tensor& tensor) {
  using Shape = extract_shape_t<Tensor>;
  constexpr int N = Shape::total_size;
  using Array = FArray<N>;
  return reinterpret_cast<const Array&>(tensor);
}

template <concepts::FTensor Tensor>
auto& reinterpret_as_array(Tensor& tensor) {
  using Shape = extract_shape_t<Tensor>;
  constexpr int N = Shape::total_size;
  using Array = FArray<N>;
  return reinterpret_cast<Array&>(tensor);
}

template <concepts::FTensor Tensor, concepts::FArray Array>
const Tensor& reinterpret_as_tensor(const Array& array) {
  static_assert(extract_length_v<Array> == extract_shape_t<Tensor>::total_size);
  return reinterpret_cast<const Tensor&>(array);
}

template <concepts::FTensor Tensor, concepts::FArray Array>
Tensor& reinterpret_as_tensor(Array& array) {
  static_assert(extract_length_v<Array> == extract_shape_t<Tensor>::total_size);
  return reinterpret_cast<Tensor&>(array);
}

template <concepts::FTensor Tensor>
float sum(const Tensor& tensor) {
  eigen_util::FTensor<Eigen::Sizes<>> out = tensor.sum();
  return out(0);
}

template <concepts::FTensor Tensor>
float max(const Tensor& tensor) {
  eigen_util::FTensor<Eigen::Sizes<>> out = tensor.maximum();
  return out(0);
}

template <concepts::FTensor Tensor>
float min(const Tensor& tensor) {
  eigen_util::FTensor<Eigen::Sizes<>> out = tensor.minimum();
  return out(0);
}

template <concepts::FTensor Tensor>
bool any(const Tensor& tensor) {
  const auto* data = tensor.data();
  for (int i = 0; i < tensor.size(); ++i) {
    if (data[i]) return true;
  }
  return false;
}

template <concepts::FTensor Tensor>
int count(const Tensor& tensor) {
  int c = 0;
  for (int i = 0; i < tensor.size(); ++i) {
    c += bool(tensor.data()[i]);
  }
  return c;
}

template <concepts::FArray Array>
void left_rotate(Array& array, int n) {
  constexpr int N = extract_length_v<Array>;
  auto* data = array.data();
  std::rotate(data, data + n, data + N);
}

template <concepts::FArray Array>
void right_rotate(Array& array, int n) {
  constexpr int N = extract_length_v<Array>;
  auto* data = array.data();
  std::rotate(data, data + N - n, data + N);
}

template <concepts::FTensor Tensor>
uint64_t hash(const Tensor& tensor) {
  using Scalar = typename Tensor::Scalar;
  constexpr int N = extract_shape_t<Tensor>::total_size;
  return util::hash_memory<N * sizeof(Scalar)>(tensor.data());
}

}  // namespace eigen_util
