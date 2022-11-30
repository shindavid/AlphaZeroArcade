#include <util/EigenUtil.hpp>

#include <array>
#include <cstdint>

namespace eigen_util {

template<typename T, typename TensorOp> T fixed_op_to_scalar(const TensorOp& op)
{
  using ScalarTensor = fixed_tensor_t<T, Eigen::Sizes<>>;
  ScalarTensor s = op;
  return s(0);
}

/*
 * Note: this implementation is a bit sensitive due to a bug in Eigen 3.4.0. See:
 *
 * https://stackoverflow.com/a/74158117/543913
 */
template<FixedTensorConcept Tensor> auto softmax(const Tensor& tensor)
{
  using T = typename Tensor::Scalar;
  auto normalized_tensor = tensor - fixed_op_to_scalar<T>(tensor.maximum());
  auto z = normalized_tensor.exp().eval();
  return z / fixed_op_to_scalar<T>(z.sum());
}

template<typename T, typename S>
torch::Tensor eigen2torch(fixed_tensor_t<T, S>& tensor)
{
  return torch::from_blob(tensor.data(), to_int64_std_array_v<S>);
}

}  // namespace eigen_util
