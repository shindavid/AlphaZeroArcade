#include <util/EigenUtil.hpp>

#include <array>
#include <cstdint>

namespace eigen_util {

template<typename T, typename S>
torch::Tensor eigen2torch(const fixed_tensor_t<T, S>& tensor)
{
  using Sizes = to_sizes_t<S>;
  auto sizes = to_int64_std_array_v<Sizes>;
  return torch::from_blob(tensor.data(), sizes);
}

}  // namespace eigen_util
