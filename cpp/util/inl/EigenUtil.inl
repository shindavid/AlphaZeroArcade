#include <util/EigenUtil.hpp>

#include <util/Random.hpp>

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

template<FixedTensorConcept Tensor>
size_t serialize(char* buf, size_t buf_size, const Tensor& tensor) {
  size_t n_bytes = sizeof(typename Tensor::Scalar) * tensor.size();
  if (n_bytes > buf_size) {
    throw util::Exception("Buffer too small (%ld > %ld)", n_bytes, buf_size);
  }
  memcpy(buf, tensor.data(), n_bytes);
  return n_bytes;
}

template<FixedTensorConcept Tensor>
void deserialize(const char* buf, Tensor* tensor) {
  memcpy(tensor->data(), buf, sizeof(typename Tensor::Scalar) * tensor->size());
}

template<typename Array> auto softmax(const Array& array) {
  auto normalized_array = array - array.maxCoeff();
  auto z = normalized_array.exp();
  return z / z.sum();
}

template<FixedTensorConcept Tensor> auto reverse(const Tensor& tensor, int dim) {
  using Sizes = extract_shape_t<Tensor>;
  constexpr int N = Sizes::count;
  static_assert(N > 0);

  Eigen::array<bool, N> rev;
  rev.fill(false);
  rev[dim] = true;
  return tensor.reverse(rev);
}

template<FixedTensorConcept Tensor>
auto sample(const Tensor& tensor) {
  using Shape = extract_shape_t<Tensor>;
  constexpr size_t N = Shape::total_size;

  const auto* data = tensor.data();
  int flat_index = util::Random::weighted_sample(data, data + N);
  return unflatten_index(tensor, flat_index);
}

template<FixedTensorConcept Tensor>
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

template<typename Scalar, ShapeConcept Shape, int Options>
const auto& reinterpret_as_array(const Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor) {
  constexpr int N = Shape::total_size;
  using ArrayT = Eigen::Array<Scalar, N, 1>;
  return reinterpret_cast<const ArrayT&>(tensor);
}

template<typename Scalar, ShapeConcept Shape, int Options>
auto& reinterpret_as_array(Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor) {
  constexpr int N = Shape::total_size;
  using ArrayT = Eigen::Array<Scalar, N, 1>;
  return reinterpret_cast<ArrayT&>(tensor);
}

template<FixedTensorConcept TensorT, typename Scalar, int N>
const TensorT& reinterpret_as_tensor(const Eigen::Array<Scalar, N, 1>& array) {
  static_assert(std::is_same_v<typename TensorT::Scalar, Scalar>);
  static_assert(N == extract_shape_t<TensorT>::total_size);
  static_assert(TensorT::Options | Eigen::RowMajor);
  return reinterpret_cast<const TensorT&>(array);
}

template<FixedTensorConcept TensorT, typename Scalar, int N>
TensorT& reinterpret_as_tensor(Eigen::Array<Scalar, N, 1>& array) {
  static_assert(std::is_same_v<typename TensorT::Scalar, Scalar>);
  static_assert(N == extract_shape_t<TensorT>::total_size);
  static_assert(TensorT::Options | Eigen::RowMajor);
  return reinterpret_cast<TensorT&>(array);
}

template<FixedMatrixConcept MatrixT, typename Scalar, ShapeConcept Shape, int Options>
const MatrixT& reinterpret_as_matrix(const Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor) {
  static_assert(std::is_same_v<typename MatrixT::Scalar, Scalar>);
  static_assert(MatrixT::RowsAtCompileTime * MatrixT::ColsAtCompileTime == Shape::total_size);
  static_assert((Options | Eigen::RowMajorBit) == (MatrixT::Flags & Eigen::RowMajorBit));
  return reinterpret_cast<const MatrixT&>(tensor);
}

template<FixedMatrixConcept MatrixT, typename Scalar, ShapeConcept Shape, int Options>
MatrixT& reinterpret_as_matrix(Eigen::TensorFixedSize<Scalar, Shape, Options>& tensor) {
  static_assert(std::is_same_v<typename MatrixT::Scalar, Scalar>);
  static_assert(MatrixT::RowsAtCompileTime * MatrixT::ColsAtCompileTime == Shape::total_size);
  static_assert((Options | Eigen::RowMajorBit) == (MatrixT::Flags & Eigen::RowMajorBit));
  return reinterpret_cast<MatrixT&>(tensor);
}

template<typename TensorT>
typename TensorT::Scalar sum(const TensorT& tensor) {
  using Scalar = typename TensorT::Scalar;
  static_assert(!std::is_same_v<Scalar, bool>, "use eigen_util::count() for bool tensors");
  Eigen::TensorFixedSize<Scalar, Eigen::Sizes<>, TensorT::Options> out = tensor.sum();
  return out(0);
}

template<typename TensorT>
typename TensorT::Scalar max(const TensorT& tensor) {
  using Scalar = typename TensorT::Scalar;
  Eigen::TensorFixedSize<Scalar, Eigen::Sizes<>, TensorT::Options> out = tensor.maximum();
  return out(0);
}

template<typename TensorT>
typename TensorT::Scalar min(const TensorT& tensor) {
  using Scalar = typename TensorT::Scalar;
  Eigen::TensorFixedSize<Scalar, Eigen::Sizes<>, TensorT::Options> out = tensor.minimum();
  return out(0);
}

template<typename TensorT>
bool any(const TensorT& tensor) {
  const auto* data = tensor.data();
  for (int i = 0; i < tensor.size(); ++i) {
    if (data[i]) return true;
  }
  return false;
}

template<typename TensorT>
int count(const TensorT& tensor) {
  int c = 0;
  for (int i = 0; i < tensor.size(); ++i) {
    c += bool(tensor.data()[i]);
  }
  return c;
}

template<typename Scalar, int N> void left_rotate(Eigen::Array<Scalar, N, 1>& array, int n) {
  Scalar* data = array.data();
  std::rotate(data, data + n, data + N);
}

template<typename Scalar, int N> void right_rotate(Eigen::Array<Scalar, N, 1>& array, int n) {
  Scalar* data = array.data();
  std::rotate(data, data + N - n, data + N);
}

template <FixedTensorConcept TensorT>
uint64_t hash(const TensorT& tensor) {
  using Scalar = typename TensorT::Scalar;
  constexpr int N = extract_shape_t<TensorT>::total_size;
  return util::hash_memory<N * sizeof(Scalar)>(tensor.data());
}

}  // namespace eigen_util
