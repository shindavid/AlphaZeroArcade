#include <util/EigenTorch.hpp>

#include <util/TorchUtil.hpp>

namespace eigentorch {

namespace detail {

template<typename Scalar, int Rank, int Options, size_t N>
torch::Tensor eigen2torch(Eigen::Tensor<Scalar, Rank, Options> &tensor,
                          const std::array<int64_t, N> &torch_shape)
{
  static_assert(Options & Eigen::RowMajorBit);
  return torch_util::from_blob(tensor.data(), torch_shape);
}

template<typename Scalar, int Rank, int Options>
torch::Tensor eigen2torch(Eigen::Tensor<Scalar, Rank, Options> &tensor)
{
  const auto dimensions = tensor.dimensions();
  std::array<int64_t, Rank> arr;
  for (int i = 0; i < Rank; ++i) {
    arr[i] = dimensions[i];
  }
  return eigen2torch(tensor, arr);
}

template<typename Scalar, typename Sizes, int Options, size_t N>
torch::Tensor eigen2torch(Eigen::TensorFixedSize<Scalar, Sizes, Options> &tensor,
                          const std::array<int64_t, N> &torch_shape)
{
  static_assert(Options & Eigen::RowMajorBit);
  return torch_util::from_blob(tensor.data(), torch_shape);
}

template<typename Scalar, typename Sizes, int Options>
torch::Tensor eigen2torch(Eigen::TensorFixedSize<Scalar, Sizes, Options> &tensor)
{
  return eigen2torch(tensor, eigen_util::to_int64_std_array_v<Sizes>);
}

template<typename Scalar, int Rows, int Cols, int Options, size_t N>
torch::Tensor eigen2torch(Eigen::Array<Scalar, Rows, Cols, Options> &matrix,
                          const std::array<int64_t, N> &torch_shape)
{
  static_assert((Options & Eigen::RowMajorBit) || (Cols == 1) || (Rows == 1));
  return torch_util::from_blob(matrix.data(), torch_shape);
}

template<typename Scalar, int Rows, int Cols, int Options>
torch::Tensor eigen2torch(Eigen::Array<Scalar, Rows, Cols, Options> &matrix)
{
  return eigen2torch(matrix, std::array<int64_t, 2>{Rows, Cols});
}

}  // namespace detail

template <typename Scalar_, typename Sizes_, int Options_>
TensorFixedSize<Scalar_, Sizes_, Options_>::TensorFixedSize()
: torch_tensor_(detail::eigen2torch(eigen_tensor_)) {}

template <typename Scalar_, typename Sizes_, int Options_>
template<typename IntT, size_t N>
TensorFixedSize<Scalar_, Sizes_, Options_>::TensorFixedSize(const std::array<IntT, N>& torch_shape)
: torch_tensor_(detail::eigen2torch(eigen_tensor_, util::array_cast<int64_t>(torch_shape))) {}

template <typename Scalar_, typename Sizes_, int Options_>
template<typename Sizes>
const typename TensorFixedSize<Scalar_, Sizes_, Options_>::template EigenSliceType<Sizes>&
TensorFixedSize<Scalar_, Sizes_, Options_>::eigenSlice(int row) const {
  return eigen_util::slice(eigen_tensor_, row);
}

template <typename Scalar_, typename Sizes_, int Options_>
template<typename Sizes>
typename TensorFixedSize<Scalar_, Sizes_, Options_>::template EigenSliceType<Sizes>&
TensorFixedSize<Scalar_, Sizes_, Options_>::eigenSlice(int row) {
  return eigen_util::slice(eigen_tensor_, row);
}

template <typename Scalar_, int Rank_, int Options_>
template<typename IntT, size_t N>
Tensor<Scalar_, Rank_, Options_>::Tensor(const std::array<IntT, N>& eigen_shape)
: eigen_tensor_(util::array_cast<int64_t>(eigen_shape))
, torch_tensor_(detail::eigen2torch(eigen_tensor_))
{}

template <typename Scalar_, int Rank_, int Options_>
template<typename IntT1, size_t N1, typename IntT2, size_t N2>
Tensor<Scalar_, Rank_, Options_>::Tensor(
    const std::array<IntT1, N1>& eigen_shape, const std::array<IntT2, N2>& torch_shape)
: eigen_tensor_(util::array_cast<int64_t>(eigen_shape))
, torch_tensor_(detail::eigen2torch(eigen_tensor_, util::array_cast<int64_t>(torch_shape)))
{}

template <typename Scalar_, int Rank_, int Options_>
template<typename Sizes>
const typename Tensor<Scalar_, Rank_, Options_>::template EigenSliceType<Sizes>&
Tensor<Scalar_, Rank_, Options_>::eigenSlice(int row) const {
  return eigen_util::slice<Sizes>(eigen_tensor_, row);
}

template <typename Scalar_, int Rank_, int Options_>
template<typename Sizes>
typename Tensor<Scalar_, Rank_, Options_>::template EigenSliceType<Sizes>&
Tensor<Scalar_, Rank_, Options_>::eigenSlice(int row) {
  return eigen_util::slice<Sizes>(eigen_tensor_, row);
}

}  // namespace eigentorch
