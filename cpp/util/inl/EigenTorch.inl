#pragma once

#include <util/EigenTorch.hpp>

namespace eigentorch {

template<typename Scalar, typename Sizes, int Options>
torch::Tensor eigen2torch(Eigen::TensorFixedSize<Scalar, Sizes, Options>& tensor) {
  return eigen2torch(tensor, eigen_util::to_int64_std_array_v<Sizes>);
}

template<typename Scalar, typename Sizes, int Options, size_t N>
torch::Tensor eigen2torch(Eigen::TensorFixedSize<Scalar, Sizes, Options>& tensor,
                          const std::array<int64_t, N>& torch_shape)
{
  static_assert(Options & Eigen::RowMajorBit);
  return torch::from_blob(tensor.data(), torch_shape);
}

template<typename Scalar, int Rows, int Cols, int Options>
torch::Tensor eigen2torch(Eigen::Matrix<Scalar, Rows, Cols, Options>& matrix) {
  return eigen2torch(matrix, std::array<int64_t, 2>{Rows, Cols});
}

template<typename Scalar, int Rows, int Cols, int Options, size_t N>
torch::Tensor eigen2torch(Eigen::Matrix<Scalar, Rows, Cols, Options>& matrix,
                          const std::array<int64_t, N>& torch_shape)
{
  static_assert((Options & Eigen::RowMajorBit) || (Cols == 1) || (Rows == 1));
  return torch::from_blob(matrix.data(), torch_shape);
}

}  // namespace eigentorch
