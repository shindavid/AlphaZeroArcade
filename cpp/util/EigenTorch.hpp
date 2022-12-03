#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <torch/torch.h>

#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

/*
 * Eigen and torch are two different third-party libraries that supply API's to interpret a float[] as a
 * mathematical tensor/matrix/vector of floats. The two libraries have their pros and cons:
 *
 * Reasons to prefer Eigen:
 * - torch does not allow for specifying the datatype at compile time. This adds some overhead to all operations, as
 *   they must all branch at runtime based on datatype.
 * - torch does not allow for compile-time specifying of sizes. This adds overhead to operations. More significantly,
 *   it is very hard to control memory allocations, as simple operations like (tensor1 + tensor2) will incur a
 *   dynamic allocation as it constructs the output tensor.
 * - Eigen, by contrast, allows for specifying the datatype and sizes at compile-time. This results in bare-metal
 *   operations on the float[], which is inherently more efficient, avoids unnecessary dynamic memory allocations,
 *   and more easily facilitates additional compiler optimizations like SSE instructions.
 *
 * Reasons to prefer torch:
 * - torch::Module (the type for neural networks) uses torch::Tensor for inputs/outputs.
 *
 * Is it possible to get the best of both worlds? To have one underlying float[], with c++ union-ish mechanics to
 * interpret that float[] either as an Eigen object or as a torch::Tensor, depending on the contextual needs?
 *
 * The answer is YES, and this module provides classes to do exactly that.
 *
 * eigentorch::FixedSizeTensor can be thought of as a c++ union of torch::Tensor and Eigen::TensorFixedSize.
 *
 * eigentorch::FixedSizeMatrix can be thought of as a c++ union of torch::Tensor and a fixed-sized Eigen::Matrix.
 *
 * eigentorch::FixedSizeVector is a template specialization of eigentorch::FixedSizeMatrix, where the second dimension
 * is 1 (analogously to how Eigen::Vector is a template specialization of Eigen::Matrix).
 */
namespace eigentorch {

/*
 * Reinterprets an eigen_util::fixed_tensor_t as a torch::Tensor. Can optionally pass in a new shape for the
 * torch tensor. By default, uses the same shape as the eigen tensor.
 *
 * This is NOT a copy. Modifying the outputted value will result in modifications to the inputted value.
 */
template<typename Scalar, typename Sizes, int Options>
torch::Tensor eigen2torch(Eigen::TensorFixedSize<Scalar, Sizes, Options>& tensor);

template<typename Scalar, typename Sizes, int Options, size_t N>
torch::Tensor eigen2torch(Eigen::TensorFixedSize<Scalar, Sizes, Options>& tensor,
                          const std::array<int64_t, N>& torch_shape);

/*
 * Reinterprets an Eigen::Matrix as a torch::Tensor. Can optionally pass in a new shape for the torch tensor. By
 * default, uses the same shape as the eigen matrix.
 *
 * This is NOT a copy. Modifying the outputted value will result in modifications to the inputted value.
 */
template<typename Scalar, int Rows, int Cols, int Options>
torch::Tensor eigen2torch(Eigen::Matrix<Scalar, Rows, Cols, Options>& matrix);

template<typename Scalar, int Rows, int Cols, int Options, size_t N>
torch::Tensor eigen2torch(Eigen::Matrix<Scalar, Rows, Cols, Options>& matrix,
                          const std::array<int64_t, N>& torch_shape);

/*
 * Can be thought of as a c++ union of torch::Tensor and Eigen::TensorFixedSize.
 *
 * The torch and eigen tensors will share the same datatype, specified by Scalar_, and the same shape, specified by
 * Sizes_. The torch shape can be overridden by passing it in as a constructor argument.
 *
 * The recommendation is to do all operations and manipulations via asEigen(), and to only use asTorch() when
 * absolutely needed (i.e., when using as input for neural network evaluation input/output).
 *
 * Warning: when mixing reading of t.asTorch() with writing of t.asEigen() or similar, I'm not sure if the compiler
 * will know that reordering is
 */
template <typename Scalar_, typename Sizes_, int Options_=Eigen::RowMajor>
class FixedSizeTensor {
public:
  using Scalar = Scalar_;
  using Sizes = Sizes_;
  static constexpr int Options = Options_;
  using EigenTensor = Eigen::TensorFixedSize<Scalar, Sizes, Options>;
  using TorchTensor = torch::Tensor;

  FixedSizeTensor() : torch_tensor_(eigentorch::eigen2torch(eigen_tensor_)) {}

  template<size_t N>
  FixedSizeTensor(const std::array<int64_t, N>& shape) : torch_tensor_(eigentorch::eigen2torch(eigen_tensor_, shape)) {}

  EigenTensor& asEigen() { return eigen_tensor_; }
  const EigenTensor& asEigen() const { return eigen_tensor_; }

  TorchTensor& asTorch() { return torch_tensor_; }
  const TorchTensor& asTorch() const { return torch_tensor_; }

private:
  EigenTensor eigen_tensor_;
  TorchTensor torch_tensor_;
};

/*
 * Can be thought of as a c++ union of torch::Tensor and Eigen::Matrix (with fixed shape).
 *
 * The torch tensor and eigen vector will share the same datatype, specified by Scalar_. By default, they will also
 * share the same 2-dimensional shape, specified by Rows_ and Cols_. The torch shape can be overridden by passing it in
 * as a constructor argument.
 *
 * The recommendation is to do all operations and manipulations via asEigen(), and to only use asTorch() when
 * absolutely needed (i.e., when using as input for neural network evaluation input/output).
 */
template <typename Scalar_, int Rows_, int Cols_, int Options_=Eigen::RowMajor>
class FixedSizeMatrix {
public:
  using Scalar = Scalar_;
  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;
  static constexpr int Options = Options_;
  using EigenMatrix = Eigen::Matrix<Scalar, Rows, Cols, Options>;
  using TorchTensor = torch::Tensor;

  FixedSizeMatrix() : torch_tensor_(eigentorch::eigen2torch(eigen_matrix_)) {}

  template<size_t N>
  FixedSizeMatrix(const std::array<int64_t, N>& shape) : torch_tensor_(eigentorch::eigen2torch(eigen_matrix_, shape)) {}

  EigenMatrix& asEigen() { return eigen_matrix_; }
  const EigenMatrix& asEigen() const { return eigen_matrix_; }

  TorchTensor& asTorch() { return torch_tensor_; }
  const TorchTensor& asTorch() const { return torch_tensor_; }

private:
  EigenMatrix eigen_matrix_;
  TorchTensor torch_tensor_;
};

template<typename Scalar, int Rows> using FixedSizeVector = FixedSizeMatrix<Scalar, Rows, 1>;

/*
 *********************************************************************
 * The following are equivalent:
 *
 * using S = eigen_util::fixed_tensor_t<float, Eigen::Sizes<1, 2>>;
 * using T = eigentorch::to_eigentorch_t<S>;
 *
 * and:
 *
 * using T = eigentorch::FixedSizeTensor<float, Eigen::Sizes<1, 2>>;
 *********************************************************************
 * Also, the following are equivalent:
 *
 * using S = Eigen::Matrix<float, 3, 4>;
 * using T = eigentorch::to_eigentorch_t<S>;
 *
 * and:
 *
 * using T = eigentorch::FixedSizeMatrix<float, 3, 4>;
 *********************************************************************
 * Also, the following are equivalent:
 *
 * using S = Eigen::Vector<float, 3>;
 * using T = eigentorch::to_eigentorch_t<S>;
 *
 * and:
 *
 * using T = eigentorch::FixedSizeVector<float, 3>;
 *********************************************************************
 */
template<typename T> struct to_eigentorch {};

template<typename Scalar, typename Sizes, int Options>
struct to_eigentorch<Eigen::TensorFixedSize<Scalar, Sizes, Options>> {
  using type = FixedSizeTensor<Scalar, Sizes>;
};

template<typename Scalar, int Rows, int Cols, int Options>
struct to_eigentorch<Eigen::Matrix<Scalar, Rows, Cols, Options>> {
  using type = FixedSizeMatrix<Scalar, Rows, Cols, Options>;
};

template<typename T> using to_eigentorch_t = typename to_eigentorch<T>::type;

}  // namespace eigentorch

#include <util/inl/EigenTorch.inl>
