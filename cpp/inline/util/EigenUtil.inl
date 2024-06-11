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

template <concepts::FTensor Tensor>
size_t serialize(char* buf, size_t buf_size, const Tensor& tensor) {
  size_t n_bytes = sizeof(typename Tensor::Scalar) * tensor.size();
  if (n_bytes > buf_size) {
    throw util::Exception("Buffer too small (%ld > %ld)", n_bytes, buf_size);
  }
  memcpy(buf, tensor.data(), n_bytes);
  return n_bytes;
}

template <concepts::FTensor Tensor>
void deserialize(const char* buf, Tensor* tensor) {
  memcpy(tensor->data(), buf, sizeof(typename Tensor::Scalar) * tensor->size());
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
  static_assert(Tensor::Options & Eigen::RowMajorBit, "Tensor must be row-major");
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

template <concepts::FTensor FTensor>
const auto& reinterpret_as_array(const FTensor& tensor) {
  using Shape = extract_shape_t<FTensor>;
  using Scalar = typename FTensor::Scalar;
  constexpr int N = Shape::total_size;
  using ArrayT = Eigen::Array<Scalar, N, 1>;
  return reinterpret_cast<const ArrayT&>(tensor);
}

template <concepts::FTensor FTensor>
auto& reinterpret_as_array(FTensor& tensor) {
  using Shape = extract_shape_t<FTensor>;
  using Scalar = typename FTensor::Scalar;
  constexpr int N = Shape::total_size;
  using ArrayT = Eigen::Array<Scalar, N, 1>;
  return reinterpret_cast<ArrayT&>(tensor);
}

template <concepts::FTensor FTensor, concepts::FArray FArray>
const FTensor& reinterpret_as_tensor(const FArray& array) {
  static_assert(extract_length_v<FArray> == extract_shape_t<FTensor>::total_size);
  return reinterpret_cast<const FTensor&>(array);
}

template <concepts::FTensor FTensor, concepts::FArray FArray>
FTensor& reinterpret_as_tensor(FArray& array) {
  static_assert(extract_length_v<FArray> == extract_shape_t<FTensor>::total_size);
  return reinterpret_cast<FTensor&>(array);
}

template <typename TensorT>
typename TensorT::Scalar sum(const TensorT& tensor) {
  using Scalar = typename TensorT::Scalar;
  static_assert(!std::is_same_v<Scalar, bool>, "use eigen_util::count() for bool tensors");
  Eigen::TensorFixedSize<Scalar, Eigen::Sizes<>, TensorT::Options> out = tensor.sum();
  return out(0);
}

template <typename TensorT>
typename TensorT::Scalar max(const TensorT& tensor) {
  using Scalar = typename TensorT::Scalar;
  Eigen::TensorFixedSize<Scalar, Eigen::Sizes<>, TensorT::Options> out = tensor.maximum();
  return out(0);
}

template <typename TensorT>
typename TensorT::Scalar min(const TensorT& tensor) {
  using Scalar = typename TensorT::Scalar;
  Eigen::TensorFixedSize<Scalar, Eigen::Sizes<>, TensorT::Options> out = tensor.minimum();
  return out(0);
}

template <typename TensorT>
bool any(const TensorT& tensor) {
  const auto* data = tensor.data();
  for (int i = 0; i < tensor.size(); ++i) {
    if (data[i]) return true;
  }
  return false;
}

template <typename TensorT>
int count(const TensorT& tensor) {
  int c = 0;
  for (int i = 0; i < tensor.size(); ++i) {
    c += bool(tensor.data()[i]);
  }
  return c;
}

template <typename Scalar, int N>
void left_rotate(Eigen::Array<Scalar, N, 1>& array, int n) {
  Scalar* data = array.data();
  std::rotate(data, data + n, data + N);
}

template <typename Scalar, int N>
void right_rotate(Eigen::Array<Scalar, N, 1>& array, int n) {
  Scalar* data = array.data();
  std::rotate(data, data + N - n, data + N);
}

template <concepts::FTensor TensorT>
uint64_t hash(const TensorT& tensor) {
  using Scalar = typename TensorT::Scalar;
  constexpr int N = extract_shape_t<TensorT>::total_size;
  return util::hash_memory<N * sizeof(Scalar)>(tensor.data());
}

}  // namespace eigen_util
